import datetime as dt

import numpy as np
import pandas as pd
import pytest

from equity_research.data import schema
from research_core.data.db import schema as core_schema
from research_core.data.db.upsert import check_fits, coerce_to_table, python_value, records


def test_missing_values_become_none_and_scalars_become_python() -> None:
    assert python_value(np.nan) is None
    assert python_value(pd.NaT) is None
    assert python_value(pd.NA) is None
    assert python_value(np.int64(3)) == 3 and isinstance(python_value(np.int64(3)), int)
    assert python_value(pd.Timestamp("2024-01-02 03:04:05")) == dt.datetime(2024, 1, 2, 3, 4, 5)


def test_records_keep_column_order() -> None:
    frame = pd.DataFrame({"b": [1.5, np.nan], "a": ["x", "y"]})
    assert records(frame, ["a", "b"]) == [("x", 1.5), ("y", None)]


def test_integer_columns_with_gaps_stay_integers() -> None:
    frame = pd.DataFrame({"issuer_id": [4.0, np.nan], "name": ["a", "b"], "listed_on": ["2020-01-02", None]})
    out = coerce_to_table(frame, core_schema.listing)
    assert records(out, ["issuer_id", "listed_on"]) == [(4, dt.date(2020, 1, 2)), (None, None)]


def test_a_value_wider_than_its_column_is_named_before_anything_is_sent() -> None:
    with pytest.raises(ValueError, match=r"equity.financial_fact\.period_type holds 3 characters: 'YTDX'"):
        check_fits(pd.DataFrame({"period_type": ["Q", "YTDX"]}), schema.financial_fact)
    with pytest.raises(ValueError, match=r"equity.filing\.fiscal_year holds -32768\.\.32767: 43101"):
        check_fits(coerce_to_table(pd.DataFrame({"fiscal_year": [2021, 43101]}), schema.filing), schema.filing)
    check_fits(pd.DataFrame({"period_type": ["YTD", None], "issuer_id": [2**31 - 1, 1]}), schema.financial_fact)
