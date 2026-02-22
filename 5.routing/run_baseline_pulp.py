"""第5章 配送計画: PuLP の実行結果を取得（検証用ベースライン）
※ ルート列挙に数分かかります。"""
import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import pulp
from itertools import product
from joblib import Parallel, delayed

# --- パラメータ（routing.ipynb と同一）---
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
o = 0
K_minus_o = K[1:]
_K = np.random.normal(0, mean_travel_time_to_destinations, size=(len(K), 2))
_K[o, :] = 0
t = np.array([[np.floor(np.linalg.norm(_K[k] - _K[l])) for k in K] for l in K])
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
    u = {
        kk: pulp.LpVariable(f"u_{kk}", lowBound=1, upBound=len(K) - 1)
        for kk in K_minus_o
    }
    h = pulp.LpVariable("h", lowBound=0, cat="Continuous")
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
    daily_route_prob += travel - H_regular <= h
    daily_route_prob += h <= H_max_overtime
    daily_route_prob += travel
    daily_route_prob.solve()
    route = {}
    for kk, ll in product(K, K):
        if kk != ll:
            route[kk, ll] = x[kk, ll].value()
        else:
            route[kk, ll] = 0
    return {
        "z": z,
        "route": route,
        "optimal": daily_route_prob.status == 1,
        "移動時間": travel.value(),
        "残業時間": h.value(),
    }


def enumerate_routes():
    routes = Parallel(n_jobs=4)(
        [delayed(simulate_route)(z) for z in product([0, 1], repeat=len(K))]
    )
    routes = pd.DataFrame(filter(lambda x: x is not None, routes))
    routes = routes[routes.optimal].copy()
    return routes


def is_OK(requests, routes_df, w, W, k):
    weight = sum([w[r] for r in requests])
    if weight > W:
        return False
    best_route_idx = None
    best_hours = sys.float_info.max
    for route_idx, row in routes_df.iterrows():
        if all([row.z[k[r]] == 1 for r in requests]) and row["移動時間"] < best_hours:
            best_route_idx = route_idx
            best_hours = row["移動時間"]
    if best_route_idx is None:
        return False
    return best_route_idx, best_hours


def _enumerate_feasible_schedules(requests_cands, current_idx_set, idx_to_add, res, routes_df, w, W, k, H_regular):
    idx_set_to_check = current_idx_set + [idx_to_add]
    next_idx = idx_to_add + 1
    is_next_idx_valid = next_idx < len(requests_cands)
    requests = [requests_cands[i] for i in idx_set_to_check]
    is_ok = is_OK(requests, routes_df, w, W, k)
    if is_ok:
        best_route_idx, best_hour = is_ok
        res.append({
            "requests": [requests_cands[i] for i in idx_set_to_check],
            "route_idx": best_route_idx,
            "hours": best_hour,
        })
        if is_next_idx_valid:
            _enumerate_feasible_schedules(requests_cands, idx_set_to_check, next_idx, res, routes_df, w, W, k, H_regular)
    if is_next_idx_valid:
        _enumerate_feasible_schedules(requests_cands, current_idx_set, next_idx, res, routes_df, w, W, k, H_regular)


def enumerate_feasible_schedules(d, R, d_0, d_1, routes_df, w, W, k, H_regular):
    requests_cands = [r for r in R if d_0[r] <= d <= d_1[r]]
    res = [{"requests": [], "route_idx": 0, "hours": 0}]
    _enumerate_feasible_schedules(requests_cands, [], 0, res, routes_df, w, W, k, H_regular)
    feasible_schedules_df = pd.DataFrame(res)
    feasible_schedules_df["overwork"] = (feasible_schedules_df.hours - H_regular).clip(0)
    feasible_schedules_df["requests_set"] = feasible_schedules_df.requests.apply(set)
    idx_cands = set(feasible_schedules_df.index)
    dominated_idx_set = set()
    for dominant_idx in feasible_schedules_df.index:
        for checked_idx in feasible_schedules_df.index:
            requests_strict_dominance = (
                feasible_schedules_df.requests_set.loc[checked_idx]
                < feasible_schedules_df.requests_set.loc[dominant_idx]
            )
            overwork_weak_dominance = (
                feasible_schedules_df.overwork.loc[checked_idx]
                >= feasible_schedules_df.overwork.loc[dominant_idx]
            )
            if requests_strict_dominance and overwork_weak_dominance:
                dominated_idx_set.add(checked_idx)
    nondominated_idx_set = idx_cands - dominated_idx_set
    return feasible_schedules_df.loc[nondominated_idx_set, :]


def main():
    print("Enumerating routes (this may take several minutes)...")
    routes_df = enumerate_routes()
    print("Enumerating feasible schedules per day...")
    _schedules = Parallel(n_jobs=4)(
        [
            delayed(enumerate_feasible_schedules)(d, R, d_0, d_1, routes_df, w, W, k, H_regular)
            for d in D
        ]
    )
    feasible_schedules = dict(zip(D, _schedules))

    prob = pulp.LpProblem(sense=pulp.LpMinimize)
    z = {}
    for d in D:
        for q in feasible_schedules[d].index:
            z[d, q] = pulp.LpVariable(f"z_{d}_{q}", cat="Binary")
    y = {r: pulp.LpVariable(f"y_{r}", cat="Continuous", lowBound=0, upBound=1) for r in R}
    deliv_count = {r: pulp.LpAffineExpression() for r in R}
    for d in D:
        for q in feasible_schedules[d].index:
            for r in feasible_schedules[d].loc[q].requests:
                deliv_count[r] += z[d, q]
    h = {
        d: pulp.lpSum(
            z[d, q] * feasible_schedules[d].overwork.loc[q]
            for q in feasible_schedules[d].index
        )
        for d in D
    }
    for d in D:
        prob += pulp.lpSum(z[d, q] for q in feasible_schedules[d].index) == 1
    for r in R:
        prob += y[r] >= 1 - deliv_count[r]
    obj_overtime = pulp.lpSum([c * h[d] for d in D])
    obj_outsourcing = pulp.lpSum([f[r] * y[r] for r in R])
    prob += obj_overtime + obj_outsourcing
    prob.solve()

    obj_val = pulp.value(prob.objective) if prob.objective is not None else None
    if obj_val is None:
        obj_val = 0.0
    result = {
        "status": prob.status,
        "objective": float(obj_val),
        "nvars": len(prob.variables()),
        "ncons": len(prob.constraints),
    }
    out_path = "baseline_pulp.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Saved", out_path)
    print(result)
    return result


if __name__ == "__main__":
    main()
