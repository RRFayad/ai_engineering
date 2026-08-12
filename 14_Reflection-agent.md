# LangGraph — Reflection Agent

## Core Idea

A **Reflection Agent** iteratively improves its own output:

1. **Generate** an initial response.
2. **Reflect** on the response and produce critique/recommendations.
3. **Generate again**, using the critique.
4. Repeat until a **stopping condition** is reached.

Flow:

`GENERATE → REFLECT → GENERATE → ... → END`

LangGraph is useful here because it explicitly orchestrates the **state, loop, and stopping condition**.

## State

The graph state stores the conversation history:

```python
class MessageGraph(TypedDict):
messages: Annotated[list[BaseMessage], add_messages]
```

`Annotated` defines the field's type plus metadata that LangGraph can interpret.

`add_messages` is the **reducer** for `messages`: instead of replacing the existing messages whenever a node returns new ones, LangGraph merges/appends them.

Conceptually:

`messages = add_messages(old_messages, new_messages)`

Without `add_messages`, each node can overwrite the previous messages, preventing the conversation history from accumulating.

## Nodes

### Generation Node

Receives the complete message history and generates/revises the answer.

Returns an `AIMessage`.

### Reflection Node

Receives the history and critiques the generated answer.

The reflection is intentionally returned as a `HumanMessage`:

```python
return {"messages": [HumanMessage(content=res.content)]}
```

Although another LLM generated the critique, `HumanMessage` makes the generator interpret it as **feedback/instructions from the other side of the conversation**.

The history therefore looks like:

`Human request → AI answer → Human critique → AI improved answer`

## Conditional Edges

After generation, a routing function decides what happens next:

```python
def should_continue(state):
if stopping_condition:
return END

    return REFLECT

```

This creates the loop:

`GENERATE → should_continue → REFLECT → GENERATE`

or:

`GENERATE → should_continue → END`

## Important: Loops & Safety

Agentic loops can trigger repeated **paid LLM calls**.

Always have:

1. A logical stopping condition.
2. A hard execution limit as a fallback.

```python
graph.invoke(
input,
config={"recursion_limit": 10}
)
```

The recursion limit protects against bugs that would otherwise cause the graph to execute indefinitely.

## Key Takeaway

LangGraph models AI workflows as:

**State + Nodes + Edges + Conditional Edges + Reducers**

For reflection:

`Generate → Critique → Improve → Evaluate whether to continue`

The important advantage over manually writing `while`/`if` logic is that LangGraph provides an explicit abstraction for **stateful workflows with branching and loops**.
