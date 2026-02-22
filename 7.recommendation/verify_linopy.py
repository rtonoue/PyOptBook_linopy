"""第7章 商品推薦: linopy + HiGHS (QP) の結果が cvxopt ベースラインと一致するか検証"""
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import xarray as xr
from linopy import Model

MIP_GAP = 1e-4
with open("baseline_cvxopt.json", encoding="utf-8") as f:
    expected = json.load(f)
with open("data_recommendation.json", encoding="utf-8") as f:
    data = json.load(f)

R, F = data["R"], data["F"]
RF2N = {(int(k.split(",")[0]), int(k.split(",")[1])): v for k, v in data["RF2N"].items()}
RF2Prob = {(int(k.split(",")[0]), int(k.split(",")[1])): v for k, v in data["RF2Prob"].items()}

Idx = [i for i in range(len(R) * len(F))]
RF2Idx = {}
for ri, r in enumerate(R):
    for fi, f in enumerate(F):
        RF2Idx[r, f] = ri * len(F) + fi

# 変数 x[dim_0] で dim_0 = 0..n-1
model = Model()
x = model.add_variables(coords=[Idx], name="x", lower=0, upper=1)

# 目的: (1/2) x' P x + q' x, P は対角, P[i,i]=2*N_i → 二次項は N_i * x_i^2
coeff_quad = xr.DataArray(
    [RF2N[R[i // len(F)], F[i % len(F)]] for i in Idx],
    dims=["dim_0"],
    coords=[Idx],
)
q_linear = xr.DataArray(
    [-2 * RF2N[R[i // len(F)], F[i % len(F)]] * RF2Prob[R[i // len(F)], F[i % len(F)]] for i in Idx],
    dims=["dim_0"],
    coords=[Idx],
)
model.add_objective((coeff_quad * x * x).sum("dim_0") + (q_linear * x).sum("dim_0"), sense="min")

# 制約 G x <= h（cvxopt と同一の制約数にするため -x<=0, x<=1 も明示）
n = len(Idx)
for i in Idx:
    g_lo = xr.DataArray([-1 if j == i else 0 for j in Idx], dims=["dim_0"], coords=[Idx])
    model.add_constraints((g_lo * x).sum("dim_0") <= 0)
for i in Idx:
    g_hi = xr.DataArray([1 if j == i else 0 for j in Idx], dims=["dim_0"], coords=[Idx])
    model.add_constraints((g_hi * x).sum("dim_0") <= 1)
# -x[r,f]+x[r+1,f] <= 0
for r in R[:-1]:
    for f in F:
        i1, i2 = RF2Idx[r, f], RF2Idx[r + 1, f]
        g = xr.DataArray([-1 if j == i1 else (1 if j == i2 else 0) for j in Idx], dims=["dim_0"], coords=[Idx])
        model.add_constraints((g * x).sum("dim_0") <= 0)
# x[r,f]-x[r,f+1] <= 0
for r in R:
    for f in F[:-1]:
        i1, i2 = RF2Idx[r, f], RF2Idx[r, f + 1]
        g = xr.DataArray([1 if j == i1 else (-1 if j == i2 else 0) for j in Idx], dims=["dim_0"], coords=[Idx])
        model.add_constraints((g * x).sum("dim_0") <= 0)

model.solve(solver_name="highs")

obj = model.objective.value
nvars = model.variables.nvars
ncons = model.constraints.ncons
assert abs(obj - expected["objective"]) <= MIP_GAP, f"objective {obj} != {expected['objective']}"
assert nvars == expected["nvars"], f"nvars {nvars} != {expected['nvars']}"
assert ncons == expected["ncons"], f"ncons {ncons} != {expected['ncons']}"
x_linopy = x.solution.values
for i in Idx:
    assert abs(x_linopy[i] - expected["x"][i]) <= 0.01, f"x[{i}] {x_linopy[i]} != {expected['x'][i]}"
print("Recommendation QP: OK", "obj=", obj, "nvars=", nvars, "ncons=", ncons)
print("検証に合格しました。")
