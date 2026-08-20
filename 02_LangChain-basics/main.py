import os
from urllib import response

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from constants import information, summary_template

load_dotenv()


def main():

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    # llm = ChatOpenAI(temperature=0, model="gpt-5") # Replaced by ollama for studying purposes
    llm = ChatOllama(temperature=0, model="gpt-oss:latest")

    # LCEL (Langchain Expression Language -> format the template and pass it to the llm)
    chain = summary_prompt_template | llm

    # We run the chain using the invoke method, passing the input information as a dictionary.
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
