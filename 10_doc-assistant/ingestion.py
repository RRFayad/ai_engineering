import os
import ssl
import asyncio
import certifi
from dotenv import load_dotenv
from typing import Any, Dict, List

from langchain_chroma import Chroma  # local alternative for Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document  # text document with metadata
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter


from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()
# ---------------- Initialization----------------------

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,  # when embedding a big list, send in to OpenAI in batches of 50
    retry_min_seconds=10,
)

# chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings). #Commented since we are using pinecone
vectorstore = PineconeVectorStore(
    embedding=embeddings,
    index_name=os.environ.get("PINECONE_INDEX_NAME"),
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

# ----------------------------------------------------------------


async def main():
    """Main async function to orchestrate the entire process"""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info(
        "TavilyCrawl: Starting to crawl docs from https://python.langchain.com",
        Colors.PURPLE,
    )

    # ---------------- STEP 1: SCRAPE AND LOAD DOCS ----------------------

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com",
            "max_depth": 5,  # Usually start with 1 - 2 and test (check docs)
            "extract_depth": "advanced",  # Extracts more data
            # "instructions": "content on ai agents",  # Natural language to instruct the crawler
        }
    )
    # For each result we want to create a LangChain Document
    all_docs = [
        (
            Document(
                page_content=result["raw_content"], metadata={"source": result["url"]}
            )
        )
        for result in res["results"]
    ]
    log_success(f"Tavily Crawl: Successfully crawled {len(all_docs)} URLs")

    # ----------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(main())
