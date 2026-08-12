from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState, add_messages

from chains import generate_chain, reflection_chain


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # critique: Annotated[str, "The critique of the tweet"]
    # new_tweet: Annotated[str, "The new tweet generated based on the critique"]


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph) -> MessageGraph:
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph) -> MessageGraph:
    # Important: Add reflection messages as Human Messages
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


def should_continue(state: MessageGraph) -> str:
    # Just to simulate a stopping condition, we will stop after 3 messages.
    if len(state["messages"]) >= 3:
        return END
    return REFLECT


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)
builder.add_conditional_edges(
    GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT}
)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().print_ascii())


if __name__ == "__main__":
    print("Hello from 14-reflection-agent!")

    input: MessageGraph = {
        "messages": [
            HumanMessage(
                content="What you need to know about bjj ranking system: A blue belt, is a good white belt. A purple belt is a black belt who doesnt have enough time on the mat yet."
            )
        ]
    }

    result = graph.invoke(input)
    print(result)
