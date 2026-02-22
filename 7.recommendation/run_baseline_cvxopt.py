"""第7章 商品推薦: cvxopt QP の実行結果を取得（検証用ベースライン）"""
import datetime
import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

try:
    import cvxopt
except ImportError:
    print("Required: cvxopt (pip install cvxopt)", file=sys.stderr)
    sys.exit(1)


def load_and_prepare():
    log_df = pd.read_csv("access_log.csv", parse_dates=["date"])
    start_date = datetime.datetime(2015, 7, 1)
    end_date = datetime.datetime(2015, 7, 7)
    target_date = datetime.datetime(2015, 7, 8)
    x_df = log_df[(start_date <= log_df["date"]) & (log_df["date"] <= end_date)]
    y_df = log_df[log_df["date"] == target_date].drop_duplicates()
    y_df["pv_flag"] = 1

    U2I2Rcens = {}
    for row in x_df.itertuples():
        rcen = (target_date - row.date).days
        U2I2Rcens.setdefault(row.user_id, {})
        U2I2Rcens[row.user_id].setdefault(row.item_id, [])
        U2I2Rcens[row.user_id][row.item_id].append(rcen)

    Rows1 = []
    for user_id, I2Rcens in U2I2Rcens.items():
        for item_id, Rcens in I2Rcens.items():
            freq = len(Rcens)
            rcen = min(Rcens)
            Rows1.append((user_id, item_id, rcen, freq))
    UI2RF_df = pd.DataFrame(Rows1, columns=["user_id", "item_id", "rcen", "freq"])
    UI2RFP_df = pd.merge(
        UI2RF_df,
        y_df[["user_id", "item_id", "pv_flag"]],
        how="left",
        on=["user_id", "item_id"],
    )
    UI2RFP_df["pv_flag"] = UI2RFP_df["pv_flag"].fillna(0)
    tar_df = UI2RFP_df[UI2RFP_df["freq"] <= 7]

    RF2N = {}
    RF2PV = {}
    for row in tar_df.itertuples():
        RF2N.setdefault((row.rcen, row.freq), 0)
        RF2PV.setdefault((row.rcen, row.freq), 0)
        RF2N[row.rcen, row.freq] += 1
        if row.pv_flag == 1:
            RF2PV[row.rcen, row.freq] += 1
    RF2Prob = {rf: RF2PV[rf] / N for rf, N in RF2N.items()}

    R = sorted(tar_df["rcen"].unique().tolist())
    F = sorted(tar_df["freq"].unique().tolist())
    Idx = []
    RF2Idx = {}
    for r in R:
        for f in F:
            idx = len(Idx)
            Idx.append(idx)
            RF2Idx[r, f] = idx
    return R, F, RF2N, RF2Prob, Idx, RF2Idx


def build_qp(R, F, RF2N, RF2Prob, Idx, RF2Idx):
    n = len(Idx)
    var_vec = [0.0] * n
    G_list = []
    h_list = []
    for r in R:
        for f in F:
            idx = RF2Idx[r, f]
            G_list.append([-1 if i == idx else 0 for i in range(n)])
            h_list.append(0)
    for r in R:
        for f in F:
            idx = RF2Idx[r, f]
            G_list.append([1 if i == idx else 0 for i in range(n)])
            h_list.append(1)
    for r in R[:-1]:
        for f in F:
            idx1, idx2 = RF2Idx[r, f], RF2Idx[r + 1, f]
            G_list.append([-1 if i == idx1 else (1 if i == idx2 else 0) for i in range(n)])
            h_list.append(0)
    for r in R:
        for f in F[:-1]:
            idx1, idx2 = RF2Idx[r, f], RF2Idx[r, f + 1]
            G_list.append([1 if i == idx1 else (-1 if i == idx2 else 0) for i in range(n)])
            h_list.append(0)

    P_list = []
    q_list = []
    for r in R:
        for f in F:
            idx = RF2Idx[r, f]
            N = RF2N[r, f]
            prob = RF2Prob[r, f]
            P_list.append([2 * N if i == idx else 0 for i in range(n)])
            q_list.append(-2 * N * prob)

    return np.array(G_list), np.array(h_list, dtype=float), np.array(P_list), np.array(q_list, dtype=float)


def main():
    R, F, RF2N, RF2Prob, Idx, RF2Idx = load_and_prepare()
    G, h, P, q = build_qp(R, F, RF2N, RF2Prob, Idx, RF2Idx)
    G_cvx = cvxopt.matrix(G, tc="d")
    h_cvx = cvxopt.matrix(h, tc="d")
    P_cvx = cvxopt.matrix(P, tc="d")
    q_cvx = cvxopt.matrix(q, tc="d")
    sol = cvxopt.solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    x_sol = np.array(sol["x"]).flatten()
    obj_val = float(sol["primal objective"])
    nvars = len(Idx)
    ncons = len(h)

    result = {
        "status": sol["status"],
        "objective": obj_val,
        "nvars": nvars,
        "ncons": ncons,
        "x": x_sol.tolist(),
    }
    with open("baseline_cvxopt.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    # 検証用に R, F, RF2N, RF2Prob を保存（キーは "r,f" 文字列）
    data = {
        "R": R,
        "F": F,
        "RF2N": {f"{r},{f}": RF2N[r, f] for r in R for f in F},
        "RF2Prob": {f"{r},{f}": RF2Prob[r, f] for r in R for f in F},
    }
    with open("data_recommendation.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("Saved baseline_cvxopt.json, data_recommendation.json")
    print("objective:", obj_val, "nvars:", nvars, "ncons:", ncons)
    return result


if __name__ == "__main__":
    main()
