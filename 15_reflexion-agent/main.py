from dotenv import load_dotenv

load_dotenv()

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executor import execute_tools

DRAFT = "draft"
EXECUTE_TOOLS = "execute_tools"
REVISE = "revise"
MAX_ITERATIONS = 2


def draft_node(state: MessagesState) -> MessagesState:
    """Draft the initial response"""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def revise_node(state: MessagesState) -> MessagesState:
    """Revise the answer based on tool results"""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState) -> Literal[EXECUTE_TOOLS, END]:  # type: ignore[valid-type]
    """Determine whether to continue or end based on iteration count (which is based on tool calls)"""
    count_tool_calls = sum(isinstance(item, ToolMessage) for item in state["messages"])

    if count_tool_calls > MAX_ITERATIONS:
        return END
    return EXECUTE_TOOLS


builder = StateGraph(MessagesState)
builder.add_node(DRAFT, draft_node)
builder.add_node(EXECUTE_TOOLS, execute_tools)
builder.add_node(REVISE, revise_node)

builder.add_edge(START, DRAFT)
builder.add_edge(DRAFT, EXECUTE_TOOLS)
builder.add_edge(EXECUTE_TOOLS, REVISE)
builder.add_conditional_edges(REVISE, event_loop, [EXECUTE_TOOLS, END])

graph = builder.compile()

print(graph.get_graph().draw_mermaid())

res = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="What are the most common presure guard passing methods in bjj?"
            )
        ]
    }
)

# Extract the final answer from the last message with tool calls
last_message = res["messages"][-1]
print(last_message)
