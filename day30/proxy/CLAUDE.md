# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload

# Test the endpoint manually
curl http://quotes.to.digital:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3.5-9b",
    "input": "напиши на asm код расчета чисел фибоначи дай только код без пояснений",
    "reasoning": "off",
    "temperature": 0.1
  }'
```

## Architecture

A minimal FastAPI proxy with a single endpoint `POST /chat` that:

1. **Rate-limits** incoming requests to 2 req/sec using a sliding window deque + `asyncio.Lock`
2. **Forwards** the request body to the upstream LLM at `https://quotes.to.digital/lm/api/v1/chat`
3. **Transforms** the JSON response — extracts the first `output[]` item with `type == "message"` and returns its `content` as plain text

Rate limit state (`request_timestamps` deque + `rate_lock`) is global per process — not shared across multiple workers.
