from langchain_core.documents import Document


def format_docs(docs: list[Document]) -> str:
    """Format the retrieved documents into a single string for the prompt context"""

    formatted = "\n\n".join([doc.page_content for doc in docs])

    return formatted
