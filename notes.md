## 1. Intro

- Objectives: Be able to develop LLM powered applications with LangChain Ecosystem (LangSmith, LangGraph)
  - Agents
  - RAGs

- **[Course Repo](https://github.com/emarco177/langchain-course)**

## 2. LangChain basics

- LangChain is an open source framework that simplifies the process of creating LLM powered apps with tools and abstractions

- One important thing is that LangChain standardize the way we consume LLMs, so it makes easy to just switch the LLM when needed

### uv notes

- We are going to use `uv` as our package manager

- `uv init`
- `uv run main.py`
- `uv add langchain`
- `uv add langchain`

### Our first project

- Installed:
  - langchain
  - langchain-openai
    - **Obs.:** LangChain decided to split these integration packages, which is good, bcs the whole package is lighter, and each provider gets the responsibility of maintaining the package
