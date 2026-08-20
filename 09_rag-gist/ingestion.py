import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()


FILE_PATH = Path(__file__).resolve().parent / "mediumblog.txt"


if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader(
        file_path=FILE_PATH,
        encoding="utf-8",
    )
    document = loader.load()

    print("Splitting...")
    # These values are heuristics basics
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(f"Created {len(texts)} chunks")

    # same dimensions from my index
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Ingesting...")
    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=os.environ.get("PINECONE_INDEX_NAME"),
    )
    print("Done!")
