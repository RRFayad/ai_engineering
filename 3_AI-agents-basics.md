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

### Creation

#### Tools

- Conceptually we need to give an agent tools and the LLM
  - A tool is a function an agent can execute
  - We need to add the decorator **`@tool`**, add the **type hinting**, and **docstrings description**, so the LLM understand when to use it

    ```python
    @tool
    def search_database(query: str, limit: int = 10) -> str:
      """
        Search customer database for record matching the query.

        Args:
          query: Search terms to look for
          limit: Max results to return
      """

      return f"Found {limit} results for '{query}'"
    ```

#### Agent

- Define the LLM, list the tools, create the agent (LLM and Tools), invoke agent

```python
  llm = ChatOllama(temperature=0, model="gpt-oss:latest")
  tools = [search_tool]
  agent = create_agent(model=llm, tools=tools)
  result = agent.invoke(
    {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}
)
```

#### Adding searching feature (with Tavily)

- Initially we added tavily inside our custom tool (regular tavily). But, tavily has its own LangChain framework, with is already ready to consume (`langchain-tavily`)

- Langchain-tavily has its tools own langchain tools, which are more precise.
  - So basically the final code is the same above, and the tool is `tools = [TavilySearch()]`
