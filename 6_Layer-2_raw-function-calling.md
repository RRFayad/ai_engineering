## 6. Layer 2 - Manual JSON Schemas vs LangChain Tool Abstraction

- Goal:
  - Understand what LangChain’s @tool and .bind_tools() actually do under the hood.
- LangChain abstractions removed:
  - @tool decorator
  - .bind_tools()
  - Automatic tool schema generation
- What we implemented manually:
  - JSON schemas describing each tool (provider-specific format, e.g. Ollama/OpenAI)
  - Passing those schemas to the LLM
  - Executing the requested tool
  - Returning the observation as a ToolMessage
  - Binding the tool result to the original request using tool_call_id
- Key takeaway:
  - Function calling is not performed by LangChain.
  - The LLM provider (OpenAI, Ollama, Anthropic, etc.) understands a tool-calling protocol and returns structured tool requests.
  - LangChain mainly abstracts the protocol and the orchestration around it.
