from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello from 16-agentic-rag!")

    res = app.invoke(input={"question": "agent memory?"})
    print(res)
    print(res["generation"])
