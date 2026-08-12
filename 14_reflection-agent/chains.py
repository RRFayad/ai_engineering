from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer, grading a tweet. Genmerate critique and recommendations for the tweet."
            "Always provide detailed recommendations, including requests for length, style, and tone.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer, generating a tweet. Generate the best tweet possible for the user request."
            "If the user provides critique, respond with a new tweet that incorporates the critique and recommendations.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI()
generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
