## 13. LangGraph

### Whats LangGraph?

- Levels of autonomy in LLM Applications:
  ![Levels of autonomy in LLM Applications](/assets/LangChain_limitations.png)

- The real autonomous are not ready for production, since they are not really reliable

- Levels:
  1. Human-driven code: Reliable, not flexible
  2. LLM Call: One step only LLM and a lot of control and some flexibility
  3. Chaining: What we learned about one output of the LLM being the next input
  4. Router: Chain that the LLM reasoning decides which branch it takes
  5. Here is where LangGraph comes - To ensure properly the iteration agentic cycles
  6. We are not really there yet

- So LangGraph is very useful for us to implement Graphs
  - We have more control with cycles, using Nodes and Edges, which pass the state through

- What are Graphs?
  - A Graph is a Mathematical object that helps us to represent relationships via nodes and edges

- State Machine
  - Are graphs where the States are Nodes and the Transitions are edges

- So we LangGraph we can describe our Nodes and Edges

### LangGraph and Flow Engineering

- Flow Engineering is an abstract concept that means a systematic and strategic approach for developing software that incorporate AI-driven decision making processes
  - So the goal is to find out and maximize it, improving the output of the AI

### LangGraph Core Components

- Components
  - Nodes
    - Python Functions (that receive states - check below)
  - Edges
    - Connect the Functions
  - Conditional Edges

- Concepts
  - **State Management**
    - A dictionary with the info to be tracked on the graph
    - So, every node receives the `state:GraphState` as argument
  - **Cyclic Graph**
    - Loops, which in LangChain they are complex to do
  - **Human In the Loop**
    - If we want to get human feedback
  - **Persistence**
