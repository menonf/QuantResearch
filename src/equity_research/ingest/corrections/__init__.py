"""Facts this repository states itself where every vendor is wrong or silent (vendor ``repository``).

    table = read_corrections()                                    # the file, checked
    ratio = distribution_ratio(parent_close, distributed_close, per_share)

``corporate_actions.csv`` lists corporate actions no vendor recorded, each with a note saying what happened:

* a **split** gives its ratio (new shares per old share);
* a **distribution** of another share line (``spinoff``: a new share class handed to holders) names the line handed
  out (``distributed_ric``) and how many of its shares come with each share (``per_share``). Its ratio is the
  holder's value on the ex-date over the parent's price alone - ``1 + per_share * distributed / parent``, both
  closes of that day - so the parent's fall is not booked as a loss.

Lines are named by RIC, never by ``listing_id`` (ids differ between databases). The loader
(:mod:`~equity_research.pipelines.load_corrections`) lands the file in the raw zone and writes the actions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PATH = Path(__file__).with_name("corporate_actions.csv")
COLUMNS = ["ric", "action_type", "ex_date", "ratio", "distributed_ric", "per_share", "note"]
ACTIONS = ("split", "spinoff")


class CorrectionError(ValueError):
    """A correction the file states badly."""


def read_corrections(path: Path = PATH) -> pd.DataFrame:
    """The corrections of ``path``, checked: known actions, a ratio for a split, a line and a count for the rest."""
    table = pd.read_csv(path, dtype={"ric": str, "distributed_ric": str, "note": str})
    if list(table.columns) != COLUMNS:
        raise CorrectionError(f"{path.name}: columns {list(table.columns)}, expected {COLUMNS}")
    table = table.assign(ex_date=pd.to_datetime(table["ex_date"]))
    problems = []
    for row in table.to_dict("records"):
        what = f"{row['ric']} {row['ex_date'].date()}"
        kind, ratio, per_share = row["action_type"], row["ratio"], row["per_share"]
        if kind not in ACTIONS:
            problems.append(f"{what}: unknown action {kind!r}")
        elif kind == "split" and not (pd.notna(ratio) and ratio > 0):
            problems.append(f"{what}: a split needs a positive ratio")
        elif kind == "spinoff" and not (
            isinstance(row["distributed_ric"], str) and pd.notna(per_share) and per_share > 0
        ):
            problems.append(f"{what}: a distribution names the line handed out and a positive count per share")
        if not (isinstance(row["note"], str) and row["note"].strip()):
            problems.append(f"{what}: every correction says what happened (note)")
    if table.duplicated(["ric", "action_type", "ex_date"]).any():
        problems.append("a correction is stated twice")
    if problems:
        raise CorrectionError(f"{path.name}: " + "; ".join(problems))
    return table


def distribution_ratio(parent_close: float, distributed_close: float, per_share: float) -> float:
    """The holder's value on the ex-date per unit of the parent's price: ``1 + per_share * distributed / parent``."""
    if not (parent_close > 0 and distributed_close > 0):
        raise CorrectionError("a distribution's ratio needs both lines' closes on its ex-date")
    return 1.0 + per_share * distributed_close / parent_close
