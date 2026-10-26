# iMac Product Environmental Report — Q&A (Complex Document RAG)

Use Case 3 of the RAG hands-on exercises: a question-answering chatbot over Apple's iMac
Product Environmental Report (October 2024, 12 pages). Built with LangChain + Google
Gemini (+ local fastembed embeddings) on the same structure as `chatbot-pilot`, with both a CLI and a browser UI.

The report is designed like a brochure. Its facts sit in tables, call-outs, a bar chart
and numbered endnotes. Text extraction gets the words but not which value belongs to
which column, or which caveat belongs to which claim. Most of the work is in ingestion.

## How it works

```
Offline (once):  PDF -> Tables rebuilt + endnotes linked + Gemini vision on visual pages
                     -> Element chunks -> local bge-small embeddings (+ BM25) -> .index/
Every question:  Question -> Hybrid search (dense + BM25, RRF), top 5 -> Cited answer
```

| Problem with basic RAG | Fix |
|---|---|
| Value separated from its column | Find each table's dark header bar in the page drawings and rebuild rows from text positions under it ("Product use: 45%" under "iMac (two ports) 256GB") |
| Caveat on a different page | Superscript markers become note references; each chunk citing a note gets that endnote attached |
| Tables cut by character count | One chunk per table; prose in small sentence groups |
| A chart tells a story the text doesn't | Gemini describes every page with images or vector artwork (445 kg baseline − 99 kg reductions = 346 kg) |

There is no reranker here. Rerankers are trained on prose and under-rate table chunks.

## Project structure

```
ingest.py       # PDF -> chunks: table rebuild, endnote linking, element chunks, visual pages
chatbot.py      # vision descriptions, embeddings, hybrid search, rag_chat()
server.py       # FastAPI bridge exposing rag_chat() to the browser UI
static/         # HTML/CSS/JS frontend (index.html, style.css, script.js)
data/           # iMac_PER_Oct2024.pdf, the source document
pyproject.toml  # dependencies (managed with uv)
```

The first run calls Gemini once per visual page, embeds every chunk, and caches the
result in `.index/` (`vision.json` holds the page descriptions). Later runs load it from
there. Delete `.index/` to rebuild.

Embeddings run locally with [fastembed](https://github.com/qdrant/fastembed)
(`BAAI/bge-small-en-v1.5`, CPU, no API quota). The model (~67 MB) downloads once on
first run. Gemini is used only for answering and describing the visual pages.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) installed
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Setup

```bash
uv sync
```

## Running it

### Option A — command-line chatbot

```bash
uv run python chatbot.py
```

You'll be prompted once for your Gemini API key (input is hidden, never saved to
disk). Chat in the terminal; type `quit` to exit.

### Option B — browser UI

```bash
uv run uvicorn server:app --reload
```

Then open **http://127.0.0.1:8000/**. The first request triggers the same API-key
prompt in the terminal running the server. Use the **Reset** button to clear
conversation memory.

To skip the prompt, export the key first: `export GOOGLE_API_KEY="your-key-here"`.

### Inspecting the chunks

```bash
uv run python ingest.py
```

Prints the rebuilt tables and the chunks carrying endnotes, without calling Gemini.

## Expected outcome

The chatbot should answer these with the right number from a chart or table, not a
general sentence nearby:

- How much less energy does iMac use compared to the ENERGY STAR requirement?
- What is the total carbon footprint of the iMac, and how much of that comes from product use versus manufacturing?
- What material is the iMac stand made from, and what percentage of it is recycled?
