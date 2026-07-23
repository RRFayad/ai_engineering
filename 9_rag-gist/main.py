import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

print("Initializing Components...")

llm = ChatOpenAI(model="gpt-3.5")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(
    embedding=embeddings,
    index_name=os.environ.get("PINECONE_INDEX_NAME"),
)

# Limit the number of retrieved documents to 3
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template("""
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Provide a Detailed Answer:
    """)


def format_docs(docs):
    """Format the retrieved documents into a single string for the prompt context"""
    return "\n\n".join([doc.page_content for doc in docs])


def retrieval_chain_without_lcel(query):
    """Retrieve documents and generate an answer without using LCEL"""

    # Step 1: Retrieve relevant documents based on the question
    docs = retriever.invoke(query)

    # Step 2: Format the retrieved documents
    context = format_docs(docs)

    # Step 3: Format the prompt with the context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Generate the answer using the LLM
    response = llm.invoke(messages)

    # Step 5: return the generated answer
    print("Answer (without LCEL):", response.content)
    return response.content


if __name__ == "__main__":
    question = "What is Pinecone in machine learning?"

    # =============================================
    # Option 0 - Without RAG
    # =============================================
    print("Generating answer without RAG...")
    result_raw = llm.invoke([HumanMessage(content=question)])
    print("Answer:")
    print(result_raw.content)

    # =============================================
    # Option 1 - Using RAG Without LCEL
    # =============================================
    print("Starting RAG process WITHOUT LCEL...")

    # =============================================
    # Option 2 - Using RAG With LCEL
    # =============================================
    print("Starting RAG process WITH LCEL...")
