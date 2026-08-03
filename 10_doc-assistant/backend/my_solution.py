import os
from typing import Any, Dict
from operator import itemgetter

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from utils.retrieval import format_docs
from utils import logger

load_dotenv()

print("Initializing Components...")

llm = ChatOpenAI()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(
    embedding=embeddings,
    index_name=os.environ.get("PINECONE_INDEX_NAME"),
)


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


prompt_template = ChatPromptTemplate.from_template("""
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, from the given context, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {query}
    Provide a Detailed Answer:
    """)


if __name__ == "__main__":
    query = "What are the main langsmith engine issue categories?"

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("query") | retriever | format_docs,
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    try:
        result = retrieval_chain.invoke({"query": query})
        logger.log_success(f"Ihaaa - {result}")

    except:
        logger.log_error("😔")
