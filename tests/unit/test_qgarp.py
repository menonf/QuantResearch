import numpy as np
import pandas as pd
import pytest

from equity_research.strategies import qgarp as qg
from equity_research.strategies.qgarp_estimates import CONSENSUS_REVISION_FACTORS, ESTIMATE_FACTORS
from equity_research.strategies.qgarp_math import growth_piecewise

DAY = pd.Timestamp("2015-06-30")
DATES = pd.DatetimeIndex([DAY])


def _frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame([values], index=DATES, columns=list(range(1, len(values) + 1)))


def test_growth_follows_productions_sign_rules() -> None:
    past = _frame([2.0, 2.0, 0.0, 0.0, -2.0, -2.0, -2.0, 2.0, np.nan])
    now = _frame([3.0, -1.0, 1.0, -1.0, 1.0, -1.0, -3.0, 0.0, 1.0])
    one_year = growth_piecewise(now, past).iloc[0].tolist()
    assert one_year[:3] == [0.5, -1.0, 1.0]  # up by half; into a loss; from zero to a profit
    assert one_year[3:5] == [-1.0, 1.5]  # from zero to a loss; from a loss to a profit: the growth itself
    assert one_year[5] == pytest.approx(0.5)  # a smaller loss: (c - p) / |p|
    assert one_year[6] == pytest.approx(-0.5)
    assert one_year[7] == -1.0 and np.isnan(one_year[8])  # profit to nothing; a missing input stays missing
    three = growth_piecewise(now, past, 3).iloc[0]
    assert three[1] == pytest.approx(1.5 ** (1 / 3) - 1)
    assert three[7] == pytest.approx(-(1.5 ** (1 / 3) - 1))  # a deeper loss compounds on its size, negated


def test_industries_follow_worldscope_groups_from_gics() -> None:
    group = pd.Series(["Banks", "Insurance", "Financial Services", None, "Transportation", "Software", None])
    sector = pd.Series(["Financials", "Financials", "Financials", "Real Estate", "Industrials", "IT", None])
    assert qg.worldscope_industry(group, sector).tolist()[:6] == [
        "BANK",
        "INSURANCE",
        "OTHER FINANCIAL",
        "OTHER FINANCIAL",
        "TRANSPORTATION",
        "INDUSTRIAL",
    ]
    assert pd.isna(qg.worldscope_industry(group, sector).iloc[6])  # nothing known: no group, not "industrial"


def test_regression_momentum_drops_missing_prices_and_counts_the_rest() -> None:
    y = pd.DataFrame({1: [0.0, 0.1, np.nan, 0.35, 0.38, 0.52]})
    t = qg._t_stats(y)[1]
    x, v = np.arange(5.0), np.array([0.0, 0.1, 0.35, 0.38, 0.52])  # x counts the prices left, as production does
    slope, intercept = np.polyfit(x, v, 1)
    resid = v - (slope * x + intercept)
    stderr = np.sqrt((resid**2).sum() / 3 / ((x - x.mean()) ** 2).sum())
    assert t == pytest.approx(slope / stderr)


def test_price_returns_need_a_recent_price() -> None:
    days = pd.bdate_range("2014-05-01", "2015-06-30")
    prices = pd.DataFrame({1: np.linspace(100.0, 150.0, len(days)), 2: 50.0}, index=days)
    prices.loc[prices.index > "2015-05-20", 2] = np.nan  # stopped trading over a month before
    out = qg.price_factors(prices, DATES)
    then = prices.loc[:"2015-05-31", 1].iloc[-1]
    assert out["1_mth_price_return_local"].loc[DAY, 1] == pytest.approx(150.0 / then - 1)
    assert out["1_year_volatility"].to_numpy(dtype=float)[0, 0] > 0
    assert out["12_mth_price_return_local"][2].isna().all() and out["price_return_stability"][2].isna().all()


def _inputs(accounts: dict[str, list[float]], *, shares: list[float], value: float) -> qg.QgarpInputs:
    """One company on one date: ``accounts[item][k]`` known ``k`` years before (and its period end)."""
    snaps = []
    for k in range(qg.SNAPSHOTS):
        names = qg.ITEMS_USED + qg.EXTRA_ITEMS
        snap = {name: _frame([accounts.get(name, [np.nan] * qg.SNAPSHOTS)[k]]) for name in names}
        end = (DAY - pd.DateOffset(years=k) - pd.Timestamp(0)).days - 90
        snap |= {f"period_end:{name}": _frame([float(end)]) for name in names}
        snaps.append(snap)
    return qg.QgarpInputs(
        dates=DATES,
        accounts=snaps,
        shares=[_frame([s]) for s in shares],
        value=[_frame([value])] * qg.SNAPSHOTS,
        dividend_yield=[_frame([0.02])] * qg.SNAPSHOTS,
        adjusted=pd.DataFrame({1: [10.0]}, index=DATES),
        industry=pd.DataFrame({1: ["INDUSTRIAL"]}, index=DATES),
    )


def test_returns_use_average_balances_and_per_share_growth_the_dates_shares() -> None:
    inputs = _inputs(
        {
            "net_income": [120.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "equity": [1000.0, 800.0, 800.0, 800.0, 800.0, 800.0],
            "total_assets": [2000.0] * 6,
        },
        shares=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        value=2400.0,
    )
    f = qg.account_factors(inputs)
    assert f["return_on_equity"].loc[DAY, 1] == pytest.approx(120.0 / 900.0)
    assert f["eps_growth_1_year_hist"].loc[DAY, 1] == pytest.approx(0.2)
    assert f["earnings_yield"].loc[DAY, 1] == pytest.approx(0.05)
    # a 2-for-1 split since a year ago doubles the count then (in today's shares): not a halving of earnings
    split = _inputs(
        {"net_income": [120.0, 100.0, 100.0, 100.0, 100.0, 100.0]},
        shares=[200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
        value=2400.0,
    )
    assert qg.account_factors(split)["eps_growth_1_year_hist"].loc[DAY, 1] == pytest.approx(0.2)


def test_a_change_between_two_snapshots_of_the_same_period_is_missing() -> None:
    profits = {"net_income": [120.0, 100.0, 90.0, 80.0, 70.0, 60.0], "total_assets": [1000.0] * 6}
    fresh = qg.account_factors(_inputs(profits, shares=[1.0] * 6, value=1.0))
    assert fresh["cash_flow_growth_1_year_change"].loc[DAY, 1] == pytest.approx(0.02)
    assert fresh["cash_flow_growth_3_year_change"].loc[DAY, 1] == pytest.approx(0.04)
    stale = _inputs(profits, shares=[1.0] * 6, value=1.0)
    stale.accounts[0]["period_end:net_income"] = stale.accounts[1]["period_end:net_income"]  # nothing new in a year
    f = qg.account_factors(stale)
    assert np.isnan(f["cash_flow_growth_1_year_change"].to_numpy(dtype=float)[0, 0])
    # two years still separate it from the third snapshot's period: at least the 700 days a 3-year change needs
    assert f["cash_flow_growth_3_year_change"].loc[DAY, 1] == pytest.approx(0.04)


def test_cleaning_sets_values_outside_limits_missing_and_leaves_out_financials() -> None:
    table = pd.DataFrame(
        {
            "date": DAY,
            "listing_id": [1, 2, 3, 4],
            "industry": ["INDUSTRIAL", "INDUSTRIAL", "BANK", "OTHER FINANCIAL"],
            "debt_to_assets": [0.5, 3.0, 0.4, 0.4],
            "book_yield": [0.1, 0.2, 0.3, 0.4],
        }
    )
    rules = pd.DataFrame(
        {
            "metric_name": ["debt_to_assets"],
            "lower_limit": [0.0],
            "upper_limit": [2.0],
            "exclude_financials": [2],
        }
    )
    out = qg.clean(table, rules)
    # outside the limits: missing, not clipped; a bank left out; other financials kept by rule 2
    assert out["debt_to_assets"].isna().tolist() == [False, True, True, False]
    assert out["debt_to_assets"].iloc[0] == 0.5
    assert out["book_yield"].tolist() == table["book_yield"].tolist()  # no rule: kept
    assert qg.clean(table) is table


def _table(n: int, **columns: list[float] | np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"date": DAY, "listing_id": range(1, n + 1), "industry": "INDUSTRIAL", **columns})


def test_growth_needs_three_of_its_four_scores_and_sustainable_growth_needs_estimates() -> None:
    rng = np.random.default_rng(0)
    from_accounts = [c for c in qg.GROWTH_INPUTS if c not in ESTIMATE_FACTORS]
    accounts_only = {c: rng.normal(size=20) for c in from_accounts}
    # as coded, the sustainable-growth score needs the NTM term; cash-flow, EPS and shareholder yield still make three
    assert qg.growth_composite(_table(20, **accounts_only))["pct_rank_growth_composite"].notna().all()
    two = {c: v for c, v in accounts_only.items() if not c.startswith("shareholder_yield")}
    assert qg.growth_composite(_table(20, **two))["pct_rank_growth_composite"].isna().all()  # two of four: none


def test_value_needs_four_yields_outside_financials_and_ranks_one_as_cheapest() -> None:
    yields = np.linspace(0.01, 0.10, 10)
    table = _table(10, earnings_yield=yields, book_yield=yields, cash_flow_yield=yields, sales_yield=yields)
    value = qg.value_composite(table)["pct_rank_value_composite"]
    assert value.iloc[-1] == 10 and value.iloc[0] == 100  # 1 = best (rank in percent: the cheapest of ten is 10)
    three = qg.value_composite(table.drop(columns="sales_yield"))["pct_rank_value_composite"]
    assert three.isna().all()  # three of six yields: production's minimum of four is not met


def test_qgarp_counts_a_missing_rank_as_the_middle() -> None:
    comp = pd.DataFrame(
        {
            "pct_rank_quality_alpha": [1.0, 50.0, 100.0, np.nan],
            "pct_rank_quality_defensive": [1.0, 50.0, 100.0, np.nan],
            "pct_rank_value_composite": [1.0, 50.0, 100.0, np.nan],
        }
    )
    rank = qg.qgarp(comp, pd.Series(DAY, index=comp.index))
    assert rank.tolist() == [25.0, 75.0, 100.0, 50.0]  # 1 = best; a name with no ranks sits in the middle
    revisions = comp.assign(zscore_earnings_revisions_1_mth=[2.0, 0.0, -2.0, 0.0])
    as_coded = qg.qgarp(revisions, pd.Series(DAY, index=comp.index))
    fixed = qg.qgarp(revisions, pd.Series(DAY, index=comp.index), fix_revisions_direction=True)
    assert as_coded.tolist() == fixed.tolist()  # the other inputs dominate here...
    level = revisions.assign(
        pct_rank_quality_alpha=50.0, pct_rank_quality_defensive=50.0, pct_rank_value_composite=50.0
    )
    gap = qg.qgarp(level, pd.Series(DAY, index=comp.index))
    assert gap.iloc[2] < gap.iloc[0]  # ...but alone, as coded, the worst revisions (-2) rank best


def test_qgarp_weights_are_configurable_and_spelled_one_way() -> None:
    assert qg.qgarp_weights(dict(qg.QGARP_WEIGHTS)) == qg.QGARP_WEIGHTS  # production's, in production's order
    momentum = {**dict(qg.QGARP_WEIGHTS), "zscore_momentum_composite": 0.2, "pct_rank_quality_defensive": 0.0}
    spelled = qg.qgarp_weights(momentum)
    assert spelled[-1] == ("zscore_momentum_composite", 0.2)  # added after production's inputs...
    assert "pct_rank_quality_defensive" not in dict(spelled)  # ...and a zero weight is no input at all
    assert qg.QgarpConfig(qgarp_weights=momentum).qgarp_weights == spelled  # one config, one registry hash
    assert qg.QgarpConfig().qgarp_weights == qg.QGARP_WEIGHTS
    for wrong in ({"momentum": 1.0}, {"pct_rank_value_composite": -0.1}, {"pct_rank_value_composite": 0.0}):
        with pytest.raises(ValueError):
            qg.qgarp_weights(wrong)


def test_qgarp_ranks_a_z_score_input_higher_is_better() -> None:
    comp = pd.DataFrame({"zscore_momentum_composite": [2.0, 0.0, -2.0], "pct_rank_value_composite": [50.0, 50.0, 50.0]})
    days = pd.Series(DAY, index=comp.index)
    rank = qg.qgarp(comp, days, weights={"zscore_momentum_composite": 1.0, "pct_rank_value_composite": 1.0})
    assert rank.tolist() == sorted(rank.tolist())  # the strongest momentum (+2) ranks best, unlike revisions as coded
    halved = qg.qgarp(comp, days, weights={"zscore_momentum_composite": 0.5, "pct_rank_value_composite": 0.5})
    assert halved.tolist() == rank.tolist()  # only the weights' ratios matter


def test_shareholder_yield_banks_and_unusual_items_follow_production() -> None:
    flows = {
        "net_income": [100.0] * 6,
        "total_assets": [1000.0] * 6,
        "operating_income": [200.0] * 6,
        "dividends_paid": [30.0, 20.0, 20.0, 10.0, 10.0, 10.0],
        "stock_repurchased": [20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "lt_debt_issued": [50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "unusual_expense": [-40.0] * 6,  # a net gain
        "loans_net": [500.0] * 6,
        "deposits": [800.0] * 6,
    }
    f = qg.account_factors(_inputs(flows, shares=[1.0] * 6, value=1000.0))
    # (30 - (0 - 20) - 50) / 1000 now, (20 - 0 - 0) / 1000 a year before: borrowing to pay holders is not a return
    assert f["shareholder_yield_growth_hist"].loc[DAY, 1] == pytest.approx(0.0 - 0.02)
    assert f["extraordinary_credit_to_operating_income"].loc[DAY, 1] == pytest.approx(0.2)
    assert f["extraordinary_charge_to_operating_income"].loc[DAY, 1] == 0.0
    assert f["loans_to_deposits"].loc[DAY, 1] == pytest.approx(-0.625)  # as coded: negated
    bank = _inputs(flows, shares=[1.0] * 6, value=1000.0)
    bank = qg.QgarpInputs(**{**bank.__dict__, "industry": pd.DataFrame({1: ["BANK"]}, index=DATES)})
    assert qg.account_factors(bank)["shareholder_yield_growth_hist"].loc[DAY, 1] == pytest.approx(0.05 - 0.02)


def test_productions_rule_set_gives_every_factor_its_two_limits() -> None:
    rules = qg.packaged_rules("gems-2026-11")
    assert rules is not None and list(rules.columns) == qg.RULE_COLUMNS
    assert len(rules) == rules["metric_name"].nunique() == 117
    assert rules[["lower_limit", "upper_limit"]].notna().all().all()
    assert (rules["lower_limit"] <= rules["upper_limit"]).all()
    assert rules["exclude_financials"].isna().all()  # this export has limits only
    by_name = rules.set_index("metric_name")
    assert list(by_name.loc["earnings_yield", ["lower_limit", "upper_limit"]]) == [-0.5, 0.5]
    assert list(by_name.loc["dividend_yield", ["lower_limit", "upper_limit"]]) == [0.0, 0.2]
    assert list(by_name.loc["capex_to_depreciation", ["lower_limit", "upper_limit"]]) == [0.0, 500.0]
    # 100 of the factors we build have limits; the rest (bank ratios, five-year volatilities...) are kept whole
    assert len(set(by_name.index) & set(qg.BUILT)) == 100
    assert qg.packaged_rules(None) is None
    with pytest.raises(ValueError, match="no rule set"):
        qg.packaged_rules("gems-1999")


def test_productions_limits_remove_only_what_lies_outside_them() -> None:
    table = _table(
        5,
        earnings_yield=[0.05, 0.5, 0.51, -0.6, np.nan],  # limits -0.5 to 0.5, inclusive
        dividend_yield=[0.0, 0.03, 0.25, -0.01, 0.2],  # 0 to 0.2
        market_capitalization=[1e3, 1e6, 1e9, 1e12, 5.0],  # no rule: kept whole
    )
    rules = qg.packaged_rules("gems-2026-11")
    out = qg.clean(table, rules)
    assert out["earnings_yield"].isna().tolist() == [False, False, True, True, True]
    assert out["dividend_yield"].isna().tolist() == [False, False, True, True, False]
    assert out["market_capitalization"].tolist() == table["market_capitalization"].tolist()

    removed = qg.removed_by_rules(table, rules)
    assert set(removed.index) == {"earnings_yield", "dividend_yield"}  # only factors with a rule
    assert list(removed.loc["earnings_yield"]) == [4, 1, 1, 0, 0.5]  # values, below, above, financials, share
    assert list(removed.loc["dividend_yield"]) == [5, 1, 1, 0, 0.4]
    assert qg.removed_by_rules(table, None).empty


def test_a_run_names_its_rule_set_and_a_run_without_one_keeps_its_old_registry_hash() -> None:
    plain = qg.registered_settings(qg.QgarpConfig())
    assert "rules" not in plain  # as stored before the setting existed: repeating that run is no new trial
    assert "benchmark" not in plain
    assert qg.registered_settings(qg.QgarpConfig(benchmark="SPY"))["benchmark"] == "SPY"
    with_rules = qg.registered_settings(qg.QgarpConfig(rules="gems-2026-11"))
    assert with_rules["rules"] == "gems-2026-11" and {k: v for k, v in with_rules.items() if k != "rules"} == plain
    with pytest.raises(ValueError, match="no rule set"):
        qg.QgarpConfig(rules="unknown")


def test_the_consensus_revisions_stand_in_for_the_revisions_and_are_ranked_as_they_are() -> None:
    table = _table(
        4, consensus_revisions_1_mth=[0.03, 0.01, -0.01, np.nan], earnings_revisions_1_mth=[0.5, 0.2, -0.1, 0.0]
    )
    comp = qg.composites(table, weights={qg.CONSENSUS_REVISIONS_1_MTH: 1.0})
    z = comp[qg.CONSENSUS_REVISIONS_1_MTH]
    assert z.iloc[0] > z.iloc[1] > z.iloc[2] and np.isnan(z.iloc[3])
    # as production ranks its revisions, ascending: the lowest revision ranks best, a missing one in the middle
    assert comp["pct_rank_qgarp"].tolist()[2] < comp["pct_rank_qgarp"].tolist()[0]
    fixed = qg.composites(table, weights={qg.CONSENSUS_REVISIONS_1_MTH: 1.0}, fix_revisions_direction=True)
    assert fixed["pct_rank_qgarp"].tolist()[0] < fixed["pct_rank_qgarp"].tolist()[2]
    assert qg.CONSENSUS_REVISIONS_1_MTH in qg.QGARP_INPUTS
    assert qg.CONSENSUS_REVISIONS_1_MTH not in dict(qg.QGARP_WEIGHTS)  # never production's


def test_factor_panels_are_one_date_by_listing_panel_per_factor() -> None:
    table = _table(3, earnings_yield=[0.05, 0.10, np.nan], pct_rank_qgarp=[1.0, 50.0, 100.0], beta=[np.nan] * 3)
    panels = qg.factor_panels(table, ["earnings_yield", "pct_rank_qgarp", "beta", "not_a_column"])
    assert set(panels) == {"earnings_yield", "pct_rank_qgarp"}  # no values or no column: left out
    assert list(panels["earnings_yield"].iloc[0])[:2] == [0.05, 0.10]
    assert list(panels["pct_rank_qgarp"].iloc[0]) == [-1.0, -50.0, -100.0]  # a composite: higher is better


def test_every_factor_and_composite_names_its_kind_of_calculation() -> None:
    """The triage table's ``calculation`` column: every score tested has a kind, every kind is explained."""
    scores = [*qg.BUILT, *qg.COMPOSITES, qg.CONSENSUS_REVISIONS_1_MTH, qg.REVISIONS_1_MTH, *CONSENSUS_REVISION_FACTORS]
    assert [n for n in scores if not qg.calculation(n)] == []
    assert set(qg.CALCULATIONS.values()) == set(qg.CALCULATION_KINDS)
    assert qg.calculation("regression_momentum_1_year") == "regression t-statistic"
    assert qg.calculation("return_on_equity_5_year_vol") == "standard deviation"
    assert qg.calculation("zscore_momentum_composite") == "z-score"
    assert qg.calculation("dividend_growth_hist") == "growth rate"
    assert qg.calculation("not_a_factor") == ""
