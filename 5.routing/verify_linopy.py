"""第5章 配送計画: linopy のメイン問題が PuLP ベースラインと一致するか検証
※ run_baseline_pulp.py と同様にルート列挙を行うため、数分かかります。"""
import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import xarray as xr
import pulp
from linopy import Model
from itertools import product
from joblib import Parallel, delayed

# run_baseline_pulp と同じパラメータ・データ
np.random.seed(10)
num_places = 10
num_days = 30
num_requests = 120
mean_travel_time_to_destinations = 100
H_regular = 8 * 60
H_max_overtime = 3 * 60
c = 3000 // 60
W = 4000
delivery_outsourcing_unit_cost = 4600
delivery_time_window = 3
avg_weight = 1000

K = list(range(num_places))
K_minus_o = K[1:]
_K = np.random.normal(0, mean_travel_time_to_destinations, size=(len(K), 2))
_K[0, :] = 0
t = np.array([[np.floor(np.linalg.norm(_K[kk] - _K[ll])) for kk in K] for ll in K])
D = list(range(num_days))
R = list(range(num_requests))
k = np.random.choice(K_minus_o, size=len(R))
d_0 = np.random.choice(D, size=len(R))
d_1 = d_0 + delivery_time_window - 1
w = np.floor(np.random.gamma(10, avg_weight / 10, size=len(R)))
f = np.ceil(w / 100) * delivery_outsourcing_unit_cost


def simulate_route(z):
    if z[0] == 0:
        return None
    daily_route_prob = pulp.LpProblem(sense=pulp.LpMinimize)
    x = {}
    for kk, ll in product(K, K):
        if kk != ll:
            x[kk, ll] = pulp.LpVariable(f"x_{kk}_{ll}", cat="Binary")
        else:
            x[kk, ll] = pulp.LpAffineExpression()
    u = {kk: pulp.LpVariable(f"u_{kk}", lowBound=1, upBound=len(K) - 1) for kk in K_minus_o}
    h_var = pulp.LpVariable("h", lowBound=0, cat="Continuous")
    for l in K:
        daily_route_prob += pulp.lpSum([x[kk, l] for kk in K]) <= 1
    for l in K:
        if z[l] == 1:
            daily_route_prob += pulp.lpSum([x[kk, l] for kk in K]) == 1
            daily_route_prob += pulp.lpSum([x[l, kk] for kk in K]) == 1
        else:
            daily_route_prob += pulp.lpSum([x[kk, l] for kk in K]) == 0
            daily_route_prob += pulp.lpSum([x[l, kk] for kk in K]) == 0
    for kk, ll in product(K_minus_o, K_minus_o):
        daily_route_prob += u[kk] + 1 <= u[ll] + len(K_minus_o) * (1 - x[kk, ll])
    travel = pulp.lpSum([t[kk, ll] * x[kk, ll] for kk, ll in product(K, K) if kk != ll])
    daily_route_prob += travel - H_regular <= h_var
    daily_route_prob += h_var <= H_max_overtime
    daily_route_prob += travel
    daily_route_prob.solve()
    route = {(kk, ll): (x[kk, ll].value() if kk != ll else 0) for kk, ll in product(K, K)}
    return {"z": z, "route": route, "optimal": daily_route_prob.status == 1, "移動時間": travel.value(), "残業時間": h_var.value()}


def enumerate_routes():
    routes = Parallel(n_jobs=4)([delayed(simulate_route)(z) for z in product([0, 1], repeat=len(K))])
    routes = pd.DataFrame(filter(lambda x: x is not None, routes))
    return routes[routes.optimal].copy()


def is_OK(requests, routes_df):
    if sum(w[r] for r in requests) > W:
        return False
    best_route_idx, best_hours = None, sys.float_info.max
    for route_idx, row in routes_df.iterrows():
        if all(row.z[k[r]] == 1 for r in requests) and row["移動時間"] < best_hours:
            best_route_idx, best_hours = route_idx, row["移動時間"]
    return (best_route_idx, best_hours) if best_route_idx is not None else False


def _enum(requests_cands, current_idx_set, idx_to_add, res, routes_df):
    idx_set_to_check = current_idx_set + [idx_to_add]
    next_idx = idx_to_add + 1
    requests = [requests_cands[i] for i in idx_set_to_check]
    is_ok = is_OK(requests, routes_df)
    if is_ok:
        best_route_idx, best_hour = is_ok
        res.append({"requests": [requests_cands[i] for i in idx_set_to_check], "route_idx": best_route_idx, "hours": best_hour})
        if next_idx < len(requests_cands):
            _enum(requests_cands, idx_set_to_check, next_idx, res, routes_df)
    if next_idx < len(requests_cands):
        _enum(requests_cands, current_idx_set, next_idx, res, routes_df)


def enumerate_feasible_schedules(d):
    requests_cands = [r for r in R if d_0[r] <= d <= d_1[r]]
    res = [{"requests": [], "route_idx": 0, "hours": 0}]
    _enum(requests_cands, [], 0, res, routes_df)
    df = pd.DataFrame(res)
    df["overwork"] = (df.hours - H_regular).clip(0)
    df["requests_set"] = df.requests.apply(set)
    dominated = set()
    for di in df.index:
        for cj in df.index:
            if df.requests_set.loc[cj] < df.requests_set.loc[di] and df.overwork.loc[cj] >= df.overwork.loc[di]:
                dominated.add(cj)
    return df.loc[set(df.index) - dominated, :]


MIP_GAP = 0.01
with open("baseline_pulp.json", encoding="utf-8") as f:
    expected = json.load(f)

print("Enumerating routes...")
routes_df = enumerate_routes()
print("Enumerating feasible schedules...")
feasible_schedules = dict(zip(D, Parallel(n_jobs=4)([delayed(enumerate_feasible_schedules)(d) for d in D])))

# メイン問題を linopy で構築（flat (d,q) インデックス）
dq_list = [(d, q) for d in D for q in feasible_schedules[d].index]
n_dq = len(dq_list)
dq_coord = list(range(n_dq))
overwork_dq = xr.DataArray(
    [feasible_schedules[d].overwork.loc[q] for d, q in dq_list],
    dims=["dq"],
    coords=[dq_coord],
)
d_for_dq = np.array([d for d, q in dq_list])
R_arr = xr.DataArray(R, dims=["r"])
A = np.zeros((len(R), n_dq))
for j, (d, q) in enumerate(dq_list):
    for r in feasible_schedules[d].loc[q].requests:
        A[r, j] = 1.0
A_xr = xr.DataArray(A, dims=["r", "dq"], coords=[R_arr, dq_coord])
f_arr = xr.DataArray(f, dims=["r"], coords=[R_arr])

model = Model()
z = model.add_variables(coords=[dq_coord], name="z", binary=True)
y = model.add_variables(coords=[R_arr], name="y", lower=0, upper=1)
for d in D:
    mask_d = xr.DataArray([1.0 if d == d_for_dq[i] else 0.0 for i in range(n_dq)], dims=["dq"], coords=[dq_coord])
    model.add_constraints((z * mask_d).sum("dq") == 1)
deliv_count = (A_xr * z).sum("dq")
model.add_constraints(y >= 1 - deliv_count)
obj_overtime = c * (z * overwork_dq).sum("dq")
obj_outsourcing = (f_arr * y).sum("r")
model.add_objective(obj_overtime + obj_outsourcing, sense="min")
model.solve(solver_name="highs")

obj = model.objective.value
nvars = model.variables.nvars
ncons = model.constraints.ncons
assert abs(obj - expected["objective"]) <= MIP_GAP, f"objective {obj} != {expected['objective']}"
assert nvars == expected["nvars"], f"nvars {nvars} != {expected['nvars']}"
assert ncons == expected["ncons"], f"ncons {ncons} != {expected['ncons']}"
print("Routing: OK", "obj=", obj, "nvars=", nvars, "ncons=", ncons)
print("検証に合格しました。")
