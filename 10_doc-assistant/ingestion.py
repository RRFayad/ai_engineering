import os
import ssl
import asyncio
import certifi
from dotenv import load_dotenv
from typing import Any, Dict, List

# ---------------- Imports (and certification) ----------------------

# Configure SSL context to use certifi certificates.
# Must run before importing langchain_tavily (aiohttp)
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_chroma import Chroma  # local alternative for Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document  # text document with metadata
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

# ---------------- Constants and Env ----------------------
load_dotenv()
URL = "https://python.langchain.com"
EXTRACT_BATCH_SIZE = 3


# ---------------- Utils ----------------------
async def extract_batch(batch_num: int, batch: List[str]) -> Dict[str, Any]:
    log_info(f"Batch {batch_num}: starting extraction of {len(batch)} URLs")
    result = await tavily_extract.ainvoke({"urls": batch, "extract_depth": "advanced"})
    log_success(f"Batch {batch_num}: finished extraction of {len(batch)} URLs")
    return result


# ---------------- Initialization----------------------

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
tavily_map = TavilyMap(max_depth=3, max_breadth=15, max_pages=500)
tavily_crawl = TavilyCrawl()

# ----------------------------------------------------------------


async def main():
    """Main async function to orchestrate the entire process"""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info(
        "TavilyCrawl: Starting to crawl docs from ",
        Colors.PURPLE,
    )

    """NOTE: Commented out step 1B since we implemented the async 1B
    # ---------------- STEP 1A: SCRAPE AND LOAD DOCS ----------------------

    # Using crawl we are mapping the URLs and Extracting content at once
    res = tavily_crawl.invoke(
        {
            "url": URL,
            "max_depth": 5,  # Usually start with 1 - 2 and test (check docs)
            "extract_depth": "advanced",  # Extracts more data
            # "instructions": "content on ai agents",  # Natural language to instruct the crawler
        }
    )
    # For each result we want to create a LangChain Document with the content and source URL
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
    """

    # ---------------- STEP 1B: SCRAPE AND LOAD DOCS WITH TAVILY EXTRACT AND TAVILY MAP (also splitting in batches) ----------------------

    log_info(
        f"TavilyMap: Mapping URLs from {URL}",
        Colors.PURPLE,
    )
    # Map the URLs
    site_map = tavily_map.invoke({"url": URL})
    urls = site_map["results"]
    log_success(f"Tavily Map: Successfully mapped {len(urls)} URLs")

    # Split into batches
    batches = [
        urls[i : i + EXTRACT_BATCH_SIZE]
        for i in range(0, len(urls), EXTRACT_BATCH_SIZE)
    ]

    # Extract each batch concurrently, instead of waiting on one single call with every URL
    extract_results = await asyncio.gather(
        *(extract_batch(i + 1, batch) for i, batch in enumerate(batches))
    )
    # Create LangChain Document with content and source, for every batch's results
    all_docs_async = [
        Document(page_content=item["raw_content"], metadata={"source": item["url"]})
        for extract_res in extract_results
        for item in extract_res["results"]
    ]
    log_success(f"Tavily Extract: Successfully extracted {len(all_docs_async)} URLs")

    # --------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(main())
