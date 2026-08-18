from typing import cast

from dotenv import load_dotenv

load_dotenv()


from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever


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
