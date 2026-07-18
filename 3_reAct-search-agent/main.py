from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv()


class Source(BaseModel):
    """Schema for a source used by the agent on the search result."""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent's response with answer and sources."""

    answer: str = Field(description="The answer from the agent")
    sources: List[Source] = Field(
        default_factory=list,
        description="The list of sources used by the agent to answer the question",
    )


llm = ChatOpenAI(temperature=0, model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


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
