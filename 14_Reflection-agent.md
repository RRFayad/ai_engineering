## 14. Reflection Agent

### Whats a Reflection Agent?

- A Reflection Agent is a general AI design pattern where a model critiques and revises its own immediate outputs inside a single loop

### Steps:

1. Setup
2. Create system reflection prompt
3. Create system generation prompt
4. Create chain

### Build Graph

- Build graph with conditional edge for reflection and generation
- **Obs.:**
  - `add_messages` is a reducer to append messages to the graph
  - We added `grandalf` to be able to use `print_ascii()`
