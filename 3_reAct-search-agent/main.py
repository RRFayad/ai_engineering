from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()


llm = ChatOllama(temperature=0, model="gpt-oss:latest")
tools = [TavilySearch()]
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
