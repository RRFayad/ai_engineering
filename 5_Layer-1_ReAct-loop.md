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
  - add prompts for the reasoning capabilities - A system message
  - Ensure there are strict defensive rules;

- **Add the Agent Loop**
  - Create the iteration where, for each iteration, we append ai messages, so:
    - We check ai message;
    - If there is no tool to call, its the final answer (also there was no content in the ai_message);
    - otherwise, we call the tool, and, append the tool calling results to the messages (thats how it gets to the LLM on the next iteration)
    - **Obs.:** The result form the tool call is bound via the `tool_id` (it broke when I removed)

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
