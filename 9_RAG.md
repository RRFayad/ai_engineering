## RAG (Retrieval Augmentation Generation) - Embeddings, Vector Databases and & Retrieval

### Concept

- Imagine we have a large document with specific info, we want the LLM use as a base
  - A very naive approach would be to give all the document context on every prompt
    - Bad, limits, tokens, latency etc
  - The other solution would be RAGs:
    - split it in chunks
    - Find the correct chunk
    - Add only the needed chunk as context in the LLM prompt

- There are some challenges on this:
  - Correct preprocessing step for generating the chunks
  - Precise search mechanism to find the relevant chunks

### Intro to Implementation

### Vector Databases

- Embeddings, Vector Stores (Pinecone), RetrievalQA Chain, LangChain Document Loaders, Document Splitters
  - Document / Document Loaders > What will contain the text (pdf, notion file etc)
  - Text Splitters > Split our text in smaller chunks, and there are different strategies for this

- We cant simply send chunks as prompt, since the more knowledge, the more expensive and redundant each api call would be
  - So we need to find out a way of sharing only the relevant chunks for a specific answer

- **Embeddings:**
  - Its a technique of creating a vector space from the text, such that the distance of the vectors in the space have a meaning
    - We can simply consider it a black box, but in theory, the vectors (and the space "position") are defined by the meaning of each snippet (word or sentence) - the closer meaning, the closer vector value

- So basically we have a book:
  - Split it into thousands or even millions of chunks > embed them using an embedding model (turning them into a vector) > save into vector database
  - When there is a query: We embed it, and then we calculate the closest vectors > closest chunks are sent in the context > LLM answer it

### Boiler Plate Setup

- Install dependencies:
  - uv add langchain langchain-community langchain_openai langchain-pinecone langchain-unstructured python-dotenv langchainhub unstructured black isort

- Pinecone:
  - Database > Create index > Text Embedding 3 Small (Open AI) > Dimension: 1536 (longer = more information)
  - Add to the env the API KEY (name it `PINECONE_API_KEY`) and also the `INDEX_NAME`
