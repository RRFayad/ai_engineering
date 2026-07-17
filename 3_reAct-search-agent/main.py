from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def search_tool(query: str) -> dict:
    """
    This is a tool that searches for information based on a query. It takes a string input (the query) and returns a string output (the search results). In this example, it simply returns a hardcoded response for demonstration purposes.
    """
    print(f"Searching for: {query}")
    response = tavily.search(query=query)
    return response


llm = ChatOllama(temperature=0, model="gpt-oss:latest")
tools = [search_tool]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from 3-react-search-agent!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Search for 3 jobs, remote, in the US, for a software engineer, that requires both React and Langchain"
                )
            ]
        }
    )

    print(result)


if __name__ == "__main__":
    main()
