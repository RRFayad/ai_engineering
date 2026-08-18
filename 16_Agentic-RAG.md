# LangGraph — Agentic RAG

[**Course Repo**](https://github.com/emarco177/langgraph-course/commits/project/agentic-rag/)

## What are we building - C-RAG (Corrective RAG)

- Basically its a more advanced RAG with reflection and more production ready
  - We get the context using RAG to retrieve Docs
  - We add a reflection layer to analyze if the retrieved docs are really relevant for the original query
    - If it is, all good
    - If it is not, filter them out, and look for more context

## Ingestion

- We are going to use ChromaDB as vector store
  - So it stores locally

- **Obs.:** The ingestion part can be improved / made more complex, but for this project we are sticking to the basics, while focusing on more complex retrieval

## State

- Simply defining the Schema
