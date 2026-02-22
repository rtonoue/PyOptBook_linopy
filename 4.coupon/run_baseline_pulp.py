"""第4章 クーポン: モデリング1（会員個別送付）の PuLP 実行結果を取得（検証用ベースライン）"""
import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
try:
    import pandas as pd
    import pulp
except ImportError:
    sys.exit(1)


def main():
    keys = ["age_cat", "freq_cat"]
    cust_df = pd.read_csv("customers.csv")
    prob_df = pd.read_csv("visit_probability.csv")
    cust_prob_df = pd.merge(cust_df, prob_df, on=keys)

    cust_prob_ver_df = cust_prob_df.rename(
        columns={"prob_dm1": 1, "prob_dm2": 2, "prob_dm3": 3}
    ).melt(id_vars=["customer_id"], value_vars=[1, 2, 3], var_name="dm", value_name="prob")
    Pim = cust_prob_ver_df.set_index(["customer_id", "dm"])["prob"].to_dict()

    I = cust_prob_df["customer_id"].to_list()
    M = [1, 2, 3]
    problem = pulp.LpProblem(name="DiscountCouponProblem1", sense=pulp.LpMaximize)
    xim = {}
    for i in I:
        for m in M:
            xim[i, m] = pulp.LpVariable(name=f"xim({i},{m})", cat="Binary")

    for i in I:
        problem += pulp.lpSum(xim[i, m] for m in M) == 1

    problem += pulp.lpSum(
        (Pim[i, m] - Pim[i, 1]) * xim[i, m] for i in I for m in [2, 3]
    )
    Cm = {1: 0, 2: 1000, 3: 2000}
    problem += pulp.lpSum(
        Cm[m] * Pim[i, m] * xim[i, m] for i in I for m in [2, 3]
    ) <= 1_000_000

    S = prob_df["segment_id"].to_list()
    Ns = cust_prob_df.groupby("segment_id")["customer_id"].count().to_dict()
    Si = cust_prob_df.set_index("customer_id")["segment_id"].to_dict()
    for s in S:
        for m in M:
            problem += pulp.lpSum(xim[i, m] for i in I if Si[i] == s) >= 0.1 * Ns[s]

    status = problem.solve()
    result = {
        "status": pulp.LpStatus[status],
        "objective": problem.objective.value(),
        "nvars": len(problem.variables()),
        "ncons": len(problem.constraints),
    }
    with open("baseline_pulp.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Saved baseline_pulp.json", result)
    return result


if __name__ == "__main__":
    main()
