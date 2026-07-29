from dotenv import load_dotenv
load_dotenv()
from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG with Tools!")
    print(app.get_graph().draw_mermaid())

    # --- Test 1: RAG question ---
    print("\n" + "="*50)
    print("TEST 1: RAG Question")
    print("="*50)
    result = app.invoke(input={"question": "what is agent memory?"})
    print("Answer:", result["generation"])

    # --- Test 2: Weather question ---
    print("\n" + "="*50)
    print("TEST 2: Weather Question")
    print("="*50)
    result = app.invoke(input={"question": "what is the weather in Hyderabad?"})
    print("Answer:", result["generation"])

    # --- Test 3: Stock question ---
    print("\n" + "="*50)
    print("TEST 3: Stock Question")
    print("="*50)
    result = app.invoke(input={"question": "what is the stock price of Tesla?"})
    print("Answer:", result["generation"])

    # --- Test 4: Calculator question ---
    print("\n" + "="*50)
    print("TEST 4: Calculator Question")
    print("="*50)
    result = app.invoke(input={"question": "calculate 25 * 48 + 100"})
    print("Answer:", result["generation"])

    # --- Test 5: Web search question ---
    print("\n" + "="*50)
    print("TEST 5: Web Search Question")
    print("="*50)
    result = app.invoke(input={"question": "what are the latest developments in LangGraph 2026?"})
    print("Answer:", result["generation"])