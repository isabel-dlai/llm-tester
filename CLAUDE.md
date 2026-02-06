# CLAUDE.md

## Project Overview

LLM Consistency Tester - a Flask app that sends the same prompt to an LLM N times in parallel and shows how consistently it responds. ChatGPT-style conversational UI.

## Quick Start

```bash
uv run app.py          # starts on http://localhost:5001
```

Requires `OPENAI_API_KEY` and optionally `ANTHROPIC_API_KEY` in `.env` (checks current dir, then parent dir).

## Architecture

Single-page Flask app. One Python file (`app.py`), one template (`templates/index.html`). No build step, no JS framework.

### Backend (app.py)

Three routes:
- `GET /` - serves the single-page template
- `POST /test` - runs the consistency test. Accepts `{ prompt, iterations, model, first_word_only }`. Fires all iterations in parallel via `asyncio.gather`, normalizes responses (lowercase, strip punctuation, optional first-word-only), returns aggregated counts + stats.
- `POST /enhance` - uses GPT-5.2 to rewrite a prompt with one-word-answer constraints. Accepts `{ prompt }`, returns `{ enhanced }`.

Model routing: models starting with `gpt-`, `o1`, `o3`, `o4` go to OpenAI; everything else goes to Anthropic.

System message enforces short (1-3 word) answers. Temperature is fixed at 1.0.

### Frontend (templates/index.html)

Self-contained HTML/CSS/JS, no external JS dependencies (Chart.js was removed). Two visual states:
- **Home state**: centered hero + chatbox + suggestion chips
- **Conversation state**: scrollable thread with user bubbles (orange, right-aligned) and assistant response bubbles (gray, left-aligned) with frequency bars

State is in-memory (`state.conversations[]` array). Settings (iterations, first-word-only) live in a popover above the chatbox. Two separate textareas for home/conversation states, synced settings.

## Key Patterns

- All API calls are async and run in parallel (`asyncio.gather`)
- Response normalization always applies: lowercase, strip punctuation, normalize whitespace. `first_word_only` is optional.
- `.env` loading checks both project dir and parent dir (for monorepo setups)
- No database, no sessions - everything is stateless per request, conversation history is client-side only
