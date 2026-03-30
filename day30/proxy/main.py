import asyncio
import time
from collections import deque

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

app = FastAPI()

RATE_LIMIT = 1  # requests per 3 seconds
LLM_URL = "https://quotes.to.digital/lm/api/v1/chat"

# Track timestamps of recent requests
request_timestamps: deque[float] = deque()
rate_lock = asyncio.Lock()


async def check_rate_limit() -> bool:
    """Returns True if request is allowed, False if rate limit exceeded."""
    async with rate_lock:
        now = time.monotonic()
        # Remove timestamps older than 1 second
        while request_timestamps and request_timestamps[0] <= now - 3.0:
            request_timestamps.popleft()

        if len(request_timestamps) >= RATE_LIMIT:
            return False

        request_timestamps.append(now)
        return True


@app.post("/chat")
async def proxy_chat(request: Request):
    if not await check_rate_limit():
        raise HTTPException(status_code=429, detail="Too Many Requests: max 1 request per 3 seconds")

    body = await request.json()

    async def do_llm_request():
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(LLM_URL, json=body)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=502, detail=f"Upstream error: {e.response.status_code}")
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"Upstream unreachable: {e}")
            return response.json()

    async def poll_disconnect():
        while not await request.is_disconnected():
            await asyncio.sleep(0.1)

    llm_task = asyncio.create_task(do_llm_request())
    disconnect_task = asyncio.create_task(poll_disconnect())

    done, pending = await asyncio.wait([llm_task, disconnect_task], return_when=asyncio.FIRST_COMPLETED)

    for task in pending:
        task.cancel()

    if disconnect_task in done:
        raise HTTPException(status_code=499, detail="Client disconnected")

    data = llm_task.result()

    outputs = data.get("output", [])
    for item in outputs:
        if item.get("type") == "message":
            return PlainTextResponse(item["content"])

    raise HTTPException(status_code=502, detail="No message content in LLM response")
