import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import TextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_unstructured import UnstructuredLoader

load_dotenv()


def main():
    print("Hello from 9-rag-gist!")
    pass


if __name__ == "__main__":
    main()
