## 1. Intro

- Objectives: Be able to develop LLM powered applications with LangChain Ecosystem (LangSmith, LangGraph)
  - Agents
  - RAGs

- **[Course Repo](https://github.com/emarco177/langchain-course)**

## 2. LangChain basics

- LangChain is an open-source framework that provides abstractions and building blocks for developing LLM-powered applications

- One important thing is that LangChain standardizes interactions with different LLM providers, making it much easier to switch between models or providers

### Our first project

- Setup:
  - Installed:
    - langchain
    - langchain-openai
      - **Obs.:** LangChain decided to split these integration packages, which is good, bcs the whole package is lighter, and each provider gets the responsibility of maintaining the package

  - `from langchain_core.prompts import PromptTemplate`
    - prompt templates are abstractions from langchain, which allows us to parametrize our prompts

  - `from langchain_openai import ChatOpenAI`
    - Wrapper over the openAi api

- We are going to write our first LangChain **chain**
  - A workflow where the output of one step is the input of the next step
  - Eg.:
    i. User Query
    ii. Prompt Template -> format query into structured prompt
    iii. LLM -> Generate response
    iv. Output parser -> Parse output into structured data
    v. Tool call -> process something (which can be an external api call)
    vi. Final LLM Call -> Process API response
    vii. Final output

- **Obs.:**
  - About temperature:
    - Low temperature (0 ~ 0.3) is more deterministic / repeatable, good for code and logic
    - High temperature (0.8+) is good for creativity, such as lyrics, poems, fiction etc

#### LCEL (Langchain Expression Language)

- A declarative syntax for composing LangChain components into chains

#### Using Ollama

- We installed and are going to use **ollama** + **gpt-oss:latest** (a light model - also its noticeable the decrease of the quality of the responses)

#### LangSmith Tracing

- Just by adding the LangSmith env vars (check docs), it tracks everything regarding the AI consumption
  - In its dashboard we can see costs, latency, token etc

## Overall Notes

### UV

- We are going to use `uv` as our package manager

- `uv init`
- `uv run main.py`
- `uv add langchain`
- `uv add langchain`

### Habits

- Always check the used framework code to understand the structure
