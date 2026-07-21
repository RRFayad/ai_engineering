## 8. Function Calling / Tool Calling

- Function Calling (or Tool Calling) is a structured protocol that allows an LLM provider to request the execution of one of the tools exposed by our application.
  - Unlike ReAct prompting, where the model mentions the desired action in free text (requiring parsing with regex or similar techniques), Function Calling returns a structured response (typically JSON).
  - The response specifies which tool to call and the arguments to pass, making the communication between the LLM and our application much more reliable and production-friendly.
