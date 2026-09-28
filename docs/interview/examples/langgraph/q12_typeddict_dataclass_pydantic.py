"""Q12. TypedDict vs dataclass vs Pydantic for state

THE PROBLEM
Choosing a state schema style for the refund desk: speed vs validation.

WHAT WE ARE GOING TO SOLVE
Run the same A100 policy step three ways - TypedDict (default), dataclass
fields as a parallel structure, and Pydantic at the trust boundary for raw input.

WHAT THIS EXAMPLE IS ABOUT
Raw web form sends days_since_delivery as a string. Pydantic coerces it.
Internal hops use TypedDict. Dataclass shown as a clean in-memory record.

WHAT IT SOLVES
TypedDict by default; Pydantic at untrusted edges; dataclass for ergonomic
internal objects (LangGraph commonly uses TypedDict/Pydantic schemas).

KEYWORDS
- TypedDict: zero-runtime-cost type hints for dict state.
- Pydantic: validates/coerces at runtime - cost vs safety.
- Trust boundary: where untrusted external data enters the graph.
"""

from dataclasses import dataclass
from typing import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph


class RefundTD(TypedDict):
    order_id: str
    days_since_delivery: int
    decision: str


class RefundIn(BaseModel):
    order_id: str = "A100"
    days_since_delivery: int = Field(ge=0)


@dataclass
class OrderRecord:
    title: str
    amount: float


def decide_td(state: RefundTD) -> dict:
    ok = state["days_since_delivery"] <= 30
    return {"decision": "approve" if ok else "deny"}


def build_typed_graph():
    g = StateGraph(RefundTD)
    g.add_node("decide_td", decide_td)
    g.add_edge(START, "decide_td")
    g.add_edge("decide_td", END)
    return g.compile()


def main() -> None:
    # Trust boundary: coerce raw form strings with Pydantic first.
    raw = {"order_id": "A100", "days_since_delivery": "12"}
    validated = RefundIn.model_validate(raw)
    record = OrderRecord(title="The Little Prince", amount=18.0)

    graph = build_typed_graph()
    out = graph.invoke(
        {
            "order_id": validated.order_id,
            "days_since_delivery": validated.days_since_delivery,
            "decision": "",
        }
    )
    print("dataclass record:", record)
    print("pydantic days:", validated.days_since_delivery, type(validated.days_since_delivery))
    print("typeddict graph decision:", out["decision"])


if __name__ == "__main__":
    print(__doc__)
    main()
