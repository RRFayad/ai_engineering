## 4. Agents under the Hood - Part 1/4

- Layer 0: LangChain abstraction - `createAgent(model=llm, tools=tools, response_format=AgentResponse)`

- Layer 1:
  - Agent Loop + Function Calling

- We are going to develop an agent for an e-commerce store
- It will have 2 tools:
  - get_product_price
  - apply_discount

### ReAct / Agent Loop

1. User Query
2. Agent Loop:
   a. Thought
   b. Action
   c. Use Tools
   d. Observes
3. Answer

- In our example:
  - system: "you are a helpful shopping assistant..."
  - user: "Whats the price of a laptop with the gold discount?"
  - Agent loop 1:
    - assistant: tool_call: get_product_price (LLM decision)
    - Our code executes and returns "1299.99"
    - Observation
  - Agent loop 2:
    - assistant: tool_call: apply_discount (LLM decision)
      - "1099.99"
    - Observation
  - Agent Loop 3:
    - No Tool to call - Give final answer

## 5. Layer 1 - The ReAct Loop

### Setup Up:

- Install dependencies:
  - `langchain`, `langchain-openai`
- Add `.env` with OpenAI and LangSmith Api Keys
- Structure file:
  - ```python
      load_dotenv()
      MAX_ITERATIONS = 10
      MODEL = "qwen3:1.7b"
    ```
- Create tools
- Create the Agent Function
  - init the llm;
  - bind tools;
  - add prompts for the reasoning capabilities
    - A system message - ENsure there are strict defensive rules;

## Overall Notes

### LangSmith

- We can add a traceable decorator to trace any function we want

```python

@traceable(
    name="agent_tool_calling",
)
def run_agent(question: str):
    pass
```
