"""Q17. Pydantic validation cost

THE PROBLEM
Validating every node hop with Pydantic slows tight retry loops on the desk bot.

WHAT WE ARE GOING TO SOLVE
Time many TypedDict updates vs Pydantic model_validate calls to show the trade.

WHAT THIS EXAMPLE IS ABOUT
Simulate 500 internal hops stamping A100 notes. Measure TypedDict dict updates
versus reconstructing a Pydantic model each time.

WHAT IT SOLVES
Pydantic costs CPU per transition - use at trust boundaries, not every hop.

KEYWORDS
- Runtime overhead: extra CPU for validation while the app runs.
- Coercion: Pydantic converting types (e.g. str -> int).
- Trust boundary: prefer validation where untrusted data enters.
"""

import time
from typing import TypedDict

from pydantic import BaseModel


class TicketTD(TypedDict):
    order_id: str
    hops: int
    note: str


class TicketPD(BaseModel):
    order_id: str
    hops: int
    note: str


def main() -> None:
    n = 500
    td: TicketTD = {"order_id": "A100", "hops": 0, "note": "The Little Prince"}
    t0 = time.perf_counter()
    for _ in range(n):
        td = {
            "order_id": td["order_id"],
            "hops": td["hops"] + 1,
            "note": td["note"],
        }
    td_ms = (time.perf_counter() - t0) * 1000

    pd = TicketPD(order_id="A100", hops=0, note="The Little Prince")
    t1 = time.perf_counter()
    for _ in range(n):
        pd = TicketPD.model_validate(
            {"order_id": pd.order_id, "hops": pd.hops + 1, "note": pd.note}
        )
    pd_ms = (time.perf_counter() - t1) * 1000

    print(f"TypedDict {n} hops: {td_ms:.2f} ms -> hops={td['hops']}")
    print(f"Pydantic  {n} hops: {pd_ms:.2f} ms -> hops={pd.hops}")
    print("Interview take: validate at edges; TypedDict for internal speed.")


if __name__ == "__main__":
    print(__doc__)
    main()
