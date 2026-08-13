from dotenv import load_dotenv

load_dotenv()

import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)

from schemas import AnswerQuestion

llm = ChatOpenAI(model="gpt-4o-mini")
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an expert researcher.
                Current time: {time}

                1. {first_instruction}
                2. Reflect and critique your own answer to the first instruction. Be severe to maximize improvement.
                3. Recommend search queries to research information and improve your answer.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)
# partial allows us to fill some variables in the prompt

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Answer the question in detail, in about 250 words."
)

if __name__ == "__main__":
    print("Hello from 15-reflexion-agent!")
    human_message = HumanMessage(
        content="In BJJ, what are the best entries for the Over Under Pass?"
    )

    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)
