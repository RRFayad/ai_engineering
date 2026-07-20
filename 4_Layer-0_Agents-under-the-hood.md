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
