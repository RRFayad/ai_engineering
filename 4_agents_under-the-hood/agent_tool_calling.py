from dotenv import load_dotenv
from langsmith import traceable
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()
MAX_ITERATIONS = 10
MODEL = "ollama:qwen3:1.7b"


# --- Tools ---
@tool
def get_product_price(product_name: str) -> float:
    """Get the price of a product."""
    print(f"Fetching price for product '{product_name}'")
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
    }
    return prices.get(product_name.lower(), 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price. Available discount tiers are 'bronze', 'silver', and 'gold'."""
    print(f"Applying discount tier '{discount_tier}' to price {price}")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount_percentage = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount_percentage / 100), 2)


# --- Agent Loop ---


@traceable(name="agent_tool_calling")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(model=MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("-" * 60)

    messages = [
        SystemMessage(
            content="You are a helpful shopping assistant. "
            "You have access to the following tools: get_product_price, apply_discount.  /n/n"
            "STRICT RULES - You must follow exactly: /n"
            "1. NEVER guess or assume any product price. You must use the get_product_price tool to fetch the price of a product. /n"
            "2. You must use the apply_discount tool to apply discounts and only AFTER you have received a price from the get_product_price tool. /n"
            "Pass the exact price returned by the get_product_price tool to the apply_discount tool. Do NOT pass a made-up number. /n"
            "3. Never calculate a discount yourself. You must use the apply_discount tool to calculate the discounted price. /n"
            "4. If the user does not specify a discount tier, ask for it - do NOT assume one. /n"
        ),
        HumanMessage(content=question),
    ]


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
