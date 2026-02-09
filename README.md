# LLM Consistency Tester

Test how consistently an LLM responds to the same prompt. Send a prompt N times in parallel, see which answers come back most often, and visualize the response distribution.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/isabelgwara/llm-tester.git
   cd llm-tester
   uv sync
   ```

2. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...   # optional, only needed for Claude models
   ```

3. Run the app:
   ```bash
   uv run app.py
   ```

4. Open http://localhost:5001

## How It Works

Type a prompt (or click a suggestion chip), choose a model from the top-right dropdown, and hit send. The app fires your prompt N times in parallel (default 10), normalizes the responses, and shows each unique answer as a chat bubble with a frequency bar.

### Features

- **30+ models** - OpenAI GPT-5/4/o-series and Anthropic Claude 4.5/4/3 families
- **Parallel execution** - all iterations run concurrently for fast results
- **Conversational UI** - ChatGPT-style interface with user/assistant bubbles
- **"+" menu** - click the + button for Enhance (rewrites your prompt for one-word answers) and Settings (iteration count, first-word-only mode)
- **Response normalization** - lowercase, strip punctuation, normalize whitespace for accurate aggregation
