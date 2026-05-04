from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.tools.pricing_tool import analyze_profit_margins as _analyze_margins
from src.tools.rag_tool import search_product_catalog as _search_catalog
from src.tools.search_tool import search_web as _search_web

load_dotenv()

MODEL_ID = "meta-llama/llama-3.3-70b-instruct"

@tool
def search_product_catalog(query: str, limit: int = 5) -> str:
    """Use this tool to answer questions about products in *our* internal
    catalog: stock levels, prices, ratings, brands, or descriptions.
    Examples: 'What wireless headphones do we have?', 'Show me AudioMax
    products'."""
    return _search_catalog(query, limit)


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Use this tool to find *current external* information: market prices,
    competitor products, industry trends, or any information not in our
    internal catalog. Examples: 'Current market price for noise-cancelling
    headphones?', 'Latest reviews for Sony WH-1000XM5'."""
    return _search_web(query, max_results)


@tool
def analyze_profit_margins(
    category: str | None = None,
    max_margin: float | None = None,
) -> str:
    """Use this tool to compute and analyze profit margins for our products
    using deterministic math. Accepts optional `category` and `max_margin`
    filters. Examples: 'Which products have the lowest profit margins?',
    'Show margins below 40% for Electronics'."""
    return _analyze_margins(category, max_margin)


SYSTEM_PROMPT = """You are a product research assistant for an e-commerce team.
Your job is to answer questions about our internal product catalog, current
external market information, and product profit margins by routing to the
right tool.

You have access to three tools:

1. `search_product_catalog` — questions about *our* internal catalog
   (stock, prices, ratings, brands, descriptions).
2. `search_web` — *current external* information (market prices, competitor
   products, industry trends, reviews).
3. `analyze_profit_margins` — deterministic margin analysis on our products,
   with optional `category` and `max_margin` filters.

Routing guidance:
- Internal catalog questions ("what do we have", "our products",
  specific brands we sell) → `search_product_catalog`.
- External / market / competitor / "current price" questions →
  `search_web`.
- Profit margin / cost / pricing-strategy math on our products →
  `analyze_profit_margins`.
- Comparative or strategic questions ("should we lower our price vs
  competitors?") usually need *multiple* tools — issue them in parallel.

Behavior rules — read carefully:
- When the answer needs data from a tool, you MUST actually invoke the tool
  via the structured tool-call channel. NEVER write a JSON
  function-call object in your text content. NEVER say "I will call X" or
  "Let's search Y" without actually issuing the call.
- Every assistant message that issues tool calls MUST also include a brief
  plain-text rationale (one or two sentences) in the message content
  explaining which tool(s) you chose and why. The rationale and the tool
  call go in the same assistant message — rationale in `content`, call in
  the tool-call channel.
- For comparative / strategic questions, issue MULTIPLE tool calls in the
  SAME assistant message (parallel tool calls), then synthesize.

Worked examples of the required format:

Example A — single tool:
  User: "What wireless headphones do we have?"
  You (one assistant message):
    content: "This is an internal catalog question, so I'll query our
    product catalog for wireless headphones."
    tool_calls: [search_product_catalog(query="wireless headphones")]

Example B — multiple parallel tools:
  User: "Should we drop the price of our Acme speaker vs competitors?"
  You (one assistant message):
    content: "I need both our internal catalog data and current market
    prices, so I'll call search_product_catalog and search_web in parallel."
    tool_calls: [
      search_product_catalog(query="Acme speaker"),
      search_web(query="Acme speaker market price competitors")
    ]

After you receive tool results, synthesize a clear, concise final answer
for the user. Cite concrete numbers and product names from the tool output
when relevant."""


tools = [search_product_catalog, search_web, analyze_profit_margins]

_OPENROUTER_KWARGS = dict(
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    extra_body={"provider": {"require_parameters": True}},
)

llm = ChatOpenAI(model=MODEL_ID, **_OPENROUTER_KWARGS).bind_tools(tools)

planner_llm = ChatOpenAI(model=MODEL_ID, **_OPENROUTER_KWARGS)

PLANNER_PROMPT = """You are the planning step of a product research agent.
The user will ask a question. In 1–3 sentences of plain text, explain how
you will approach it and which of these tools you will use:

- `search_product_catalog` — internal catalog (stock, price, brand,
  description, rating).
- `search_web` — current external info (market price, competitors,
  reviews, trends).
- `analyze_profit_margins` — profit-margin math on our products.

For comparative or strategic questions, plan to use multiple tools. Output
ONLY the rationale — do NOT call any tool, do NOT include JSON, do NOT
answer the user's question yet."""


def planner_node(state: MessagesState):
    messages = [SystemMessage(PLANNER_PROMPT)] + state["messages"]
    response = planner_llm.invoke(messages)
    return {"messages": [response]}


def agent_node(state: MessagesState):
    messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [llm.invoke(messages)]}


graph = StateGraph(MessagesState)
graph.add_node("planner", planner_node)
graph.add_node("agent", agent_node)
graph.add_node("action", ToolNode(tools))
graph.add_edge(START, "planner")
graph.add_edge("planner", "agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "action", END: END})
graph.add_edge("action", "agent")
app = graph.compile()


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if b.get("type") == "text"
        ).strip()
    return ""


def run_agent(query: str) -> dict:
    final_state = app.invoke({"messages": [HumanMessage(query)]})
    messages = final_state["messages"]

    ai_messages = [m for m in messages if isinstance(m, AIMessage)]

    tools_used: list[str] = []
    for msg in ai_messages:
        for tc in msg.tool_calls or []:
            name = tc["name"]
            if name not in tools_used:
                tools_used.append(name)

    final_ai = (
        ai_messages[-1]
        if ai_messages and not ai_messages[-1].tool_calls
        else None
    )
    answer = _extract_text(final_ai.content) if final_ai else ""

    reasoning_parts: list[str] = []
    for msg in ai_messages:
        if msg is final_ai:
            continue
        text = _extract_text(msg.content)
        if text:
            reasoning_parts.append(text)

    return {
        "answer": answer,
        "reasoning": "\n\n".join(reasoning_parts),
        "tools_used": tools_used,
    }
