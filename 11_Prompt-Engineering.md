## 11. Prompt Engineering Theory

### What is Language Modeling

- Its a probability Distribution over sequences of words - SO its the task of predicting what word will come next
  - So all autocompletes (like in the phone keyboard, google search etc) are Language Models

- So, LLMs (Large Language Models) are simply Language Models trained with a huge amount of data (basically the whole internet)

### Composition of a Prompt

- Instructions

- Context

- Input Data

- Output Indicator

### Zero, One and a Few Shot Prompts

- Zero Shots:
  - A prompt that asks for a task without any examples
  - Its accuracy might be imprecise, the more specific is the request

- One Shot

- A few Shot

### Chain of Thought Prompting

- Basically its more precise, giving an example of breaking the thought in step ny step

### ReAct Prompting

- Chain of Thought + Tools

### Prompt Engineering Tips

- Context - Be specific enough
- Clear, non ambiguous tasks
- **Iterate**
