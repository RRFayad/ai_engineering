## 7. Layer 3 - ReAct Prompt

- Goal:
  - Remove even the provider’s native function-calling support and implement the entire reasoning loop ourselves.
- LangChain abstractions removed:
  - Tool abstraction
  - Message abstraction
  - Tool execution orchestration
- Provider abstractions removed:
  - Native function calling
  - Structured tool calls
  - tool_call_id
- What we implemented manually:
  - The ReAct prompt (Thought → Action → Action Input → Observation)
  - Parsing the model’s text output using regex
  - Calling Python functions ourselves
  - Feeding the observation back into the prompt
  - Repeating the reasoning loop until a final answer is produced
- Key takeaway:
  - The model doesn’t actually “know” what a tool is.
  - It is simply continuing a text conversation following the ReAct pattern.
  - Our Python code interprets the model’s output, executes the appropriate function, and injects the observation back into the prompt.

* **LangChain Prompts:**
  - [LangChain Hub ReAct prompt structure](https://smith.langchain.com/hub/hwchase17/)
