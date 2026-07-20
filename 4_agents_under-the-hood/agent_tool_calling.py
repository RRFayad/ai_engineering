from dotenv import load_dotenv
from langsmith import traceable
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()
MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


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


@traceable(
    name="agent_tool_calling",
)
def run_agent(question: str):
    pass


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
