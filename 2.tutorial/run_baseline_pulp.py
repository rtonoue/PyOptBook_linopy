"""
現行実装（PuLP）の実行結果を取得し、JSONで保存する。
検証時に linopy 版と比較するためのベースライン。
"""
import json
import os
import sys

# PuLP が無い場合は dev-dependencies で入れる想定
try:
    import pulp
except ImportError:
    print("PuLP が必要です: uv add --dev pulp", file=sys.stderr)
    sys.exit(1)

def run_sle():
    """2.1 連立一次方程式 (SLE): 目的なし・等号制約のみ（PuLPはmaximizeで目的0）"""
    problem = pulp.LpProblem('SLE', pulp.LpMaximize)
    x = pulp.LpVariable('x', cat='Continuous')
    y = pulp.LpVariable('y', cat='Continuous')
    problem += 120 * x + 150 * y == 1440
    problem += x + y == 10
    status = problem.solve()
    obj = problem.objective.value() if problem.objective else 0.0
    return {
        "name": "SLE",
        "status": pulp.LpStatus[status],
        "objective": obj,
        "nvars": len(problem.variables()),
        "ncons": len(problem.constraints),
        "x": x.value(),
        "y": y.value(),
    }

def run_lp():
    """2.2 線形計画問題 (LP)"""
    problem = pulp.LpProblem('LP', pulp.LpMaximize)
    x = pulp.LpVariable('x', cat='Continuous')
    y = pulp.LpVariable('y', cat='Continuous')
    problem += 1 * x + 3 * y <= 30
    problem += 2 * x + 1 * y <= 40
    problem += x >= 0
    problem += y >= 0
    problem.objective = x + 2 * y
    status = problem.solve()
    return {
        "name": "LP",
        "status": pulp.LpStatus[status],
        "objective": problem.objective.value(),
        "nvars": len(problem.variables()),
        "ncons": len(problem.constraints),
        "x": x.value(),
        "y": y.value(),
    }

def run_lp2():
    """2.3 規模の大きな線形計画 (LP2)"""
    import pandas as pd
    require_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'requires.csv'))
    stock_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'stocks.csv'))
    gain_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'gains.csv'))
    P = gain_df['p'].tolist()
    M = stock_df['m'].tolist()
    stock = {row.m: row.stock for row in stock_df.itertuples()}
    gain = {row.p: row.gain for row in gain_df.itertuples()}
    require = {(row.p, row.m): row.require for row in require_df.itertuples()}

    problem = pulp.LpProblem('LP2', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', P, cat='Continuous')
    for p in P:
        problem += x[p] >= 0
    for m in M:
        problem += pulp.lpSum([require[p, m] * x[p] for p in P]) <= stock[m]
    problem += pulp.lpSum([gain[p] * x[p] for p in P])

    status = problem.solve()
    return {
        "name": "LP2",
        "status": pulp.LpStatus[status],
        "objective": problem.objective.value(),
        "nvars": len(problem.variables()),
        "ncons": len(problem.constraints),
        "x": {p: x[p].value() for p in P},
    }

def run_ip():
    """2.3 整数計画 (IP)"""
    import pandas as pd
    require_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'requires.csv'))
    stock_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'stocks.csv'))
    gain_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'gains.csv'))
    P = gain_df['p'].tolist()
    M = stock_df['m'].tolist()
    stock = {row.m: row.stock for row in stock_df.itertuples()}
    gain = {row.p: row.gain for row in gain_df.itertuples()}
    require = {(row.p, row.m): row.require for row in require_df.itertuples()}

    problem = pulp.LpProblem('IP', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', P, cat='Integer')
    for p in P:
        problem += x[p] >= 0
    for m in M:
        problem += pulp.lpSum([require[p, m] * x[p] for p in P]) <= stock[m]
    problem += pulp.lpSum([gain[p] * x[p] for p in P])

    status = problem.solve()
    return {
        "name": "IP",
        "status": pulp.LpStatus[status],
        "objective": problem.objective.value(),
        "nvars": len(problem.variables()),
        "ncons": len(problem.constraints),
        "x": {p: x[p].value() for p in P},
    }

def main():
    os.chdir(os.path.dirname(__file__))
    results = {
        "SLE": run_sle(),
        "LP": run_lp(),
        "LP2": run_lp2(),
        "IP": run_ip(),
    }
    out_path = os.path.join(os.path.dirname(__file__), 'baseline_pulp.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Baseline saved to", out_path)
    for k, v in results.items():
        print(k, "status=", v["status"], "obj=", v["objective"], "nvars=", v["nvars"], "ncons=", v["ncons"])

if __name__ == "__main__":
    main()
