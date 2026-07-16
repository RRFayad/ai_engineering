## 3. AI Agents basics

- An Agent is a Software System that takes LLMs as reasoning engine to decide what actions to take, and then executes those actions
  - So agents uses the LLM to decide, while in Chains, LLMs executes tasks, but we decide the entire flow

- In real world projects its essentially giving tools to the LLMs
  - **ReAct** architecture:
    - ReAct = Reasoning and Acting
    - Query > LLM thinking > LLM decides what tool to use (like an API call)
    - LLM receives the new info from the tool
    - Create the finished answer

- We are going to develop a job search agent

### Environment Setup

- uv init
- uv add:
  - langchain
  - langchain_openai (and langchain_ollama)
  - langchain-tavily tavily-python - Standard library used for web browsing
  - black isort

## Overall Notes
