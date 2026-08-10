from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    """
    Determine whether the agent should continue reasoning or act based on the last message.
    """

    last_message = state["messages"][LAST]

    if not isinstance(last_message, AIMessage):
        return END

    if not last_message.tool_calls:
        return END  # If there are no tool calls, we should act.

    return ACT  # If there are tool calls, we should continue reasoning.


flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {END: END, ACT: ACT})
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="langgraph_flow.png")


if __name__ == "__main__":
    print("ReAct LangGraph")
    res = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the weather in Tokyo? List it and then triple the temperature."
                )
            ]
        }
    )
    print(res["messages"][LAST].content)
