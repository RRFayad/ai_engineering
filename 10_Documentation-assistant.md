## 10. Document Assistant

- This will be a E2E complete project, considering:
  - Ingesting Documentation
  - RAG
  - LLM Memory
  - Streamlit FE

### Setup

- Pinecone:
  - OpenAI text embeddings 3 small - 1536 dimensions

- LangSmith

#### Dependencies

- **LLM / LangChain core**
  - `langchain` - core framework (chains, prompts, LCEL)
  - `langchain-openai` - OpenAI-specific integrations (chat models, embeddings)
  - `openai` - official OpenAI SDK, used under the hood by `langchain-openai`
  - `tiktoken` - OpenAI's tokenizer, used to count/estimate tokens (chunking, cost)

- **Vector store (Pinecone)**
  - `pinecone-client` - official Pinecone SDK (create/manage the index, upsert/query vectors)
  - `langchain-pinecone` - LangChain's `VectorStore` wrapper around Pinecone

- **Document ingestion / parsing**
  - `beautifulsoup4` - HTML parsing (used when scraping/cleaning the crawled LangChain docs pages)
  - `unstructured` - document loader/parser for many file types (html, pdf, etc.), used by LangChain's loaders
  - `nltk` - NLP toolkit; `unstructured` depends on it for text preprocessing (tokenizing sentences, etc.)

- **Backend / API**
  - `fastapi` - web framework to expose the assistant as an API
  - `uvicorn` - ASGI server to run the FastAPI app
  - `jinja2` - HTML templating engine (used by FastAPI for server-rendered pages, if any)

- **Frontend**
  - `streamlit` - builds the chat UI quickly, no separate frontend framework needed
  - `streamlit-chat` - chat-style message components on top of Streamlit

- **Utils / tooling**
  - `tqdm` - progress bars (useful during ingestion of many docs/chunks)
  - `python-dotenv` - loads env vars (`.env`) like API keys into the process
  - `black` - code formatter (dev tool, not runtime)

- We are going to work with `tavily-crawl`
- Its going to scrape LangChain Docs, so we can ingest it

### Imports

```python
import os
import ssl
import asyncio
import certifi
from dotenv import load_dotenv
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

- `os` - env vars
- `ssl` + `certifi` - fix HTTPS cert verification for the crawl requests
- `asyncio` - run ingestion async (crawl many pages concurrently)
- `load_dotenv` - load `.env`
- `typing` - type hints
- `Chroma` - local/alt vector store (for quick tests vs Pinecone)
- `OpenAIEmbeddings` - embed text into vectors
- `Document` - LangChain's core text + metadata object
- `PineconeVectorStore` - store/query embeddings in Pinecone
- `TavilyCrawl` / `TavilyExtract` / `TavilyMap` - scrape LangChain docs (crawl site, map URLs, extract content)
- `RecursiveCharacterTextSplitter` - split docs into chunks before embedding
