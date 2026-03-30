import asyncio
import time
from collections import deque

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

app = FastAPI()

RATE_LIMIT = 2  # requests per second
LLM_URL = "https://quotes.to.digital/lm/api/v1/chat"

# Track timestamps of recent requests
request_timestamps: deque[float] = deque()
rate_lock = asyncio.Lock()


async def check_rate_limit() -> bool:
    """Returns True if request is allowed, False if rate limit exceeded."""
    async with rate_lock:
        now = time.monotonic()
        # Remove timestamps older than 1 second
        while request_timestamps and request_timestamps[0] <= now - 1.0:
            request_timestamps.popleft()

        if len(request_timestamps) >= RATE_LIMIT:
            return False

        request_timestamps.append(now)
        return True


@app.post("/chat")
async def proxy_chat(request: Request):
    if not await check_rate_limit():
        raise HTTPException(status_code=429, detail="Too Many Requests: max 2 requests per second")

    body = await request.json()

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(LLM_URL, json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e.response.status_code}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Upstream unreachable: {e}")
        data = response.json()

    outputs = data.get("output", [])
    for item in outputs:
        if item.get("type") == "message":
            return PlainTextResponse(item["content"])

    raise HTTPException(status_code=502, detail="No message content in LLM response")
