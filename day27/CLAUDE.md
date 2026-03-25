# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses [uv](https://docs.astral.sh/uv/) for dependency management (Python >= 3.13).

```bash
# Install dependencies
uv sync

# Run: extract requisites from a document
uv run python -m src.main -f documents/yandex.docx

# Run: interactive chat mode (no file)
uv run python -m src.main

# Override LM Studio endpoint
uv run python -m src.main --url http://localhost:1234 -f documents/yandex.docx

# Save session to a named history file
uv run python -m src.main -f documents/yandex.docx -o my_session
```

There are no tests in this project.

## Architecture

The agent extracts Russian company legal requisites (ИНН, ОГРН, bank details, etc.) from documents using a local LLM via LM Studio.

**Data flow:**

1. `main.py` parses CLI args and orchestrates the two modes:
   - **Document mode** (`-f`): extracts text → sends to LLM → parses JSON response → saves to history
   - **Interactive mode**: chat loop that sends current known data as context with each user message

2. `file_extractor.py` converts `.pdf` / `.docx` / `.xlsx` / `.txt` files to plain text. `.txt` tries `utf-8`, `cp1251`, `iso-8859-5` encodings (Russian documents).

3. `llm_client.py` posts to the LM Studio OpenAI-compatible endpoint. Pass the full URL including `/v1/chat/completions` via `--url`. It has exponential-backoff retry (3 attempts) on network errors. `clean_json_output()` strips markdown code fences from model responses.

4. `models.py` defines Pydantic v2 models: `CompanyRequisites` (validates INN/OGRN digit lengths), `BankDetails` (validates 20-digit account numbers), `ConversationHistory` (wraps everything for JSON persistence).

5. `history_manager.py` persists `ConversationHistory` to `history/<timestamp>_<uuid>.json` (or a named file with `-o`). `update_requisites()` calls `merger.py` to deep-merge new LLM data into existing state, preserving fields already known.

6. `merger.py` deep-merges new LLM-extracted dicts into existing `CompanyRequisites`, handling nested `bank_details` separately. Falls back to the previous valid object if the merged result fails Pydantic validation.

**LLM model:** defaults to `qwen3.5-35b-a3b` (hardcoded in `main.py` calls to `client.request_completion`).
