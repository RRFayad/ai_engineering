from typing import cast

from dotenv import load_dotenv
from pprint import pprint

load_dotenv()


from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from ingestion import retriever
from graph.chains.router import question_router, RouteQuery


def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    print(f"Retrieved {len(docs)} docs")
    print(docs)

    doc_txt = docs[0].page_content

    res = cast(
        GradeDocuments,
        retrieval_grader.invoke({"question": question, "document": doc_txt}),
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    print(f"Retrieved {len(docs)} docs")
    print(docs)

    doc_txt = docs[1].page_content

    res = cast(
        GradeDocuments,
        retrieval_grader.invoke({"question": "how to make pizaa", "document": doc_txt}),
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test_router_to_vectorstore() -> None:
    question = "agent memory"

    res = question_router.invoke({"question": question})
    assert res.get("datasource") == "vectorstore"


def test_router_to_websearch() -> None:
    question = "how to make pizza"

    res = question_router.invoke({"question": question})
    assert res.get("datasource") == "websearch"
