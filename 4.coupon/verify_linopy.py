"""第4章 クーポン モデリング1: linopy の計算結果が PuLP ベースラインと一致するか検証"""
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import xarray as xr
from linopy import Model

MIP_GAP = 0.01  # 目的関数値は実数なので少し余裕
with open("baseline_pulp.json", encoding="utf-8") as f:
    expected = json.load(f)

keys = ["age_cat", "freq_cat"]
cust_df = pd.read_csv("customers.csv")
prob_df = pd.read_csv("visit_probability.csv")
cust_prob_df = pd.merge(cust_df, prob_df, on=keys)
cust_prob_ver_df = cust_prob_df.rename(
    columns={"prob_dm1": 1, "prob_dm2": 2, "prob_dm3": 3}
).melt(id_vars=["customer_id"], value_vars=[1, 2, 3], var_name="dm", value_name="prob")
Pim_dict = cust_prob_ver_df.set_index(["customer_id", "dm"])["prob"].to_dict()

I = cust_prob_df["customer_id"].to_list()
M = [1, 2, 3]
S = prob_df["segment_id"].to_list()
Ns = cust_prob_df.groupby("segment_id")["customer_id"].count().to_dict()
Si = cust_prob_df.set_index("customer_id")["segment_id"].to_dict()
Cm = {1: 0, 2: 1000, 3: 2000}

I_arr = xr.DataArray(I, dims=["i"])
M_arr = xr.DataArray(M, dims=["m"])
Pim = xr.DataArray(
    [[Pim_dict[i, m] for m in M] for i in I], coords=[I_arr, M_arr], dims=["i", "m"]
)

model = Model()
x = model.add_variables(coords=[I_arr, M_arr], name="x", binary=True)
model.add_constraints(x.sum("m") == 1)

# 目的: sum (Pim[i,m]-Pim[i,1])*x[i,m] for i,m with m in [2,3]
gain = Pim - Pim.sel(m=1)
gain_23 = gain.isel(m=slice(1, 3))  # m=2,3
x_23 = x.isel(m=slice(1, 3))
model.add_objective((x_23 * gain_23).sum(), sense="max")

# 予算: sum Cm[m]*Pim[i,m]*x[i,m] for i, m in [2,3] <= 1000000
cost_coef = xr.DataArray([Cm[m] for m in M], coords=[M_arr], dims=["m"])
cost_23 = (cost_coef * Pim * x).isel(m=slice(1, 3)).sum()
model.add_constraints(cost_23 <= 1_000_000)

# セグメント別 10% 以上
for s in S:
    for m in M:
        mask_i = xr.DataArray([1 if Si[i] == s else 0 for i in I], coords=[I_arr], dims=["i"])
        model.add_constraints((x.sel(m=m) * mask_i).sum("i") >= 0.1 * Ns[s])

model.solve(solver_name="highs")

obj = model.objective.value
nvars = model.variables.nvars
ncons = model.constraints.ncons
assert abs(obj - expected["objective"]) <= MIP_GAP, f"objective {obj} != {expected['objective']}"
assert nvars == expected["nvars"], f"nvars {nvars} != {expected['nvars']}"
assert ncons == expected["ncons"], f"ncons {ncons} != {expected['ncons']}"
print("Coupon Problem1: OK", "obj=", obj, "nvars=", nvars, "ncons=", ncons)
print("検証に合格しました。")
