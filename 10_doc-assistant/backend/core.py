import os
from typing import Any, Dict, TypedDict

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware.types import InputAgentState
from langchain.messages import ToolMessage
from langchain_core.messages import AnyMessage
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_pinecone import PineconeVectorStore

load_dotenv()


# -------------------------------------- Types --------------------------------------
class RAGResult(TypedDict):
    answer: str
    context: list[Document]


# -------------------------------------- Initialize --------------------------------------

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = PineconeVectorStore(
    embedding=embeddings,
    index_name=os.environ.get("PINECONE_INDEX_NAME"),
)

model = init_chat_model("openai:gpt-5.5", temperature=0)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain"""

    # Retrieve top 4 most similar docs
    retireved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get("source", "Unknown")}\n\nContent: {doc.page_content}"
        )
        for doc in retireved_docs
    )

    # return both serialized and raw docs
    return serialized, retireved_docs


def run_llm(query: str) -> Any:
    """
    Run RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: User's query

    Returns:
        Dictionary, containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """

    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build messages list (considering types)
    messages: list[AnyMessage | dict[str, Any]] = [{"role": "user", "content": query}]

    # Invoke the agent
    agent_input: InputAgentState = {"messages": messages}
    response = agent.invoke(agent_input)

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    result = run_llm(query="What are deep agents?")
    print("RESULT:", result)
