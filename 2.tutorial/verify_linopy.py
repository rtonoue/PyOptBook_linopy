"""
linopy 版の計算結果がベースライン（PuLP）と一致するか検証する。
- 最適値が MIP_GAP 以下の精度で一致すること
- 変数数・制約数が一致すること
"""
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import xarray as xr
from linopy import Model

MIP_GAP = 1e-6
expected_path = "baseline_expected.json"
with open(expected_path, encoding="utf-8") as f:
    expected = json.load(f)


def check(name, obj, nvars, ncons):
    e = expected[name]
    ok_obj = abs(obj - e["objective"]) <= MIP_GAP or (
        e["objective"] != 0 and abs((obj - e["objective"]) / e["objective"]) <= MIP_GAP
    )
    ok_nvars = nvars == e["nvars"]
    ok_ncons = ncons == e["ncons"]
    if not ok_obj:
        raise AssertionError(f"{name}: objective {obj} != expected {e['objective']}")
    if not ok_nvars:
        raise AssertionError(f"{name}: nvars {nvars} != expected {e['nvars']}")
    if not ok_ncons:
        raise AssertionError(f"{name}: ncons {ncons} != expected {e['ncons']}")
    print(f"{name}: OK (obj={obj}, nvars={nvars}, ncons={ncons})")


# データ読み込み（LP2, IP 用）
require_df = pd.read_csv("requires.csv")
stock_df = pd.read_csv("stocks.csv")
gain_df = pd.read_csv("gains.csv")
P = gain_df["p"].tolist()
M = stock_df["m"].tolist()
stock = {row.m: row.stock for row in stock_df.itertuples()}
gain = {row.p: row.gain for row in gain_df.itertuples()}
require = {(row.p, row.m): row.require for row in require_df.itertuples()}
require_arr = xr.DataArray(
    [[require[p, m] for m in M] for p in P], coords=[P, M], dims=["p", "m"]
)
gain_arr = xr.DataArray([gain[p] for p in P], coords=[P], dims=["p"])

# SLE
m_sle = Model()
x_sle = m_sle.add_variables(name="x")
y_sle = m_sle.add_variables(name="y")
m_sle.add_constraints(120 * x_sle + 150 * y_sle == 1440)
m_sle.add_constraints(x_sle + y_sle == 10)
m_sle.add_objective(0 * x_sle)  # 連立方程式なので目的は定数0（linopyは式のみ受け付ける）
m_sle.solve(solver_name="highs")
check("SLE", m_sle.objective.value, m_sle.variables.nvars, m_sle.constraints.ncons)
# 解の値も確認
assert abs(float(x_sle.solution) - 2.0) < 1e-5 and abs(float(y_sle.solution) - 8.0) < 1e-5

# LP
m_lp = Model()
x_lp = m_lp.add_variables(lower=0, name="x")
y_lp = m_lp.add_variables(lower=0, name="y")
m_lp.add_constraints(1 * x_lp + 3 * y_lp <= 30)
m_lp.add_constraints(2 * x_lp + 1 * y_lp <= 40)
m_lp.add_constraints(x_lp >= 0)
m_lp.add_constraints(y_lp >= 0)
m_lp.add_objective(x_lp + 2 * y_lp, sense="max")
m_lp.solve(solver_name="highs")
check("LP", m_lp.objective.value, m_lp.variables.nvars, m_lp.constraints.ncons)
assert abs(float(x_lp.solution) - 18.0) < 1e-5 and abs(float(y_lp.solution) - 4.0) < 1e-5

# LP2（PuLP と同様に非負制約を明示して ncons=7 に揃える）
model_lp2 = Model()
x_lp2 = model_lp2.add_variables(lower=0, coords=[xr.DataArray(P, dims=["p"])], name="x")
model_lp2.add_constraints(x_lp2 >= 0)  # 4 constraints (PuLP と同様)
model_lp2.add_constraints(
    (x_lp2 * require_arr).sum("p")
    <= xr.DataArray([stock[mat] for mat in M], coords=[M], dims=["m"])
)
model_lp2.add_objective((x_lp2 * gain_arr).sum(), sense="max")
model_lp2.solve(solver_name="highs")
check("LP2", model_lp2.objective.value, model_lp2.variables.nvars, model_lp2.constraints.ncons)
# 解の値（ベースラインと一致）
baseline_lp2 = {"p1": 12.142857, "p2": 3.5714286, "p3": 7.4285714, "p4": 0.0}
for p in P:
    assert abs(float(x_lp2.solution.sel(p=p)) - baseline_lp2[p]) < 1e-5

# IP（同様に非負制約を明示して ncons=7）
model_ip = Model()
x_ip = model_ip.add_variables(
    lower=0, coords=[xr.DataArray(P, dims=["p"])], name="x", integer=True
)
model_ip.add_constraints(x_ip >= 0)
model_ip.add_constraints(
    (x_ip * require_arr).sum("p")
    <= xr.DataArray([stock[mat] for mat in M], coords=[M], dims=["m"])
)
model_ip.add_objective((x_ip * gain_arr).sum(), sense="max")
model_ip.solve(solver_name="highs")
check("IP", model_ip.objective.value, model_ip.variables.nvars, model_ip.constraints.ncons)
baseline_ip = {"p1": 13, "p2": 3, "p3": 7, "p4": 0}
for p in P:
    assert abs(float(x_ip.solution.sel(p=p)) - baseline_ip[p]) < 1e-5

print("\nすべての検証に合格しました。計算結果はベースライン（PuLP）と一致しています。")
