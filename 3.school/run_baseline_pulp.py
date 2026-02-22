"""第3章 学校のクラス編成: PuLP の実行結果を取得（検証用ベースライン）"""
import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
try:
    import pandas as pd
    import pulp
except ImportError as e:
    print("Required: pandas, pulp", file=sys.stderr)
    sys.exit(1)


def main():
    s_df = pd.read_csv("students.csv")
    s_pair_df = pd.read_csv("student_pairs.csv")

    prob = pulp.LpProblem("ClassAssignmentProblem", pulp.LpMaximize)
    S = s_df["student_id"].tolist()
    C = ["A", "B", "C", "D", "E", "F", "G", "H"]
    SC = [(s, c) for s in S for c in C]
    x = pulp.LpVariable.dicts("x", SC, cat="Binary")

    for s in S:
        prob += pulp.lpSum([x[s, c] for c in C]) == 1
    for c in C:
        prob += pulp.lpSum([x[s, c] for s in S]) >= 39
        prob += pulp.lpSum([x[s, c] for s in S]) <= 40

    S_male = [row.student_id for row in s_df.itertuples() if row.gender == 1]
    S_female = [row.student_id for row in s_df.itertuples() if row.gender == 0]
    for c in C:
        prob += pulp.lpSum([x[s, c] for s in S_male]) <= 20
        prob += pulp.lpSum([x[s, c] for s in S_female]) <= 20

    score = {row.student_id: row.score for row in s_df.itertuples()}
    score_mean = s_df["score"].mean()
    for c in C:
        prob += pulp.lpSum([x[s, c] * score[s] for s in S]) >= (score_mean - 10) * pulp.lpSum([x[s, c] for s in S])
        prob += pulp.lpSum([x[s, c] * score[s] for s in S]) <= (score_mean + 10) * pulp.lpSum([x[s, c] for s in S])

    S_leader = [row.student_id for row in s_df.itertuples() if row.leader_flag == 1]
    for c in C:
        prob += pulp.lpSum([x[s, c] for s in S_leader]) >= 2

    S_support = [row.student_id for row in s_df.itertuples() if row.support_flag == 1]
    for c in C:
        prob += pulp.lpSum([x[s, c] for s in S_support]) <= 1

    SS = [(row.student_id1, row.student_id2) for row in s_pair_df.itertuples()]
    for s1, s2 in SS:
        for c in C:
            prob += x[s1, c] + x[s2, c] <= 1

    s_df = s_df.copy()
    s_df["score_rank"] = s_df["score"].rank(ascending=False, method="first")
    class_dic = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
    s_df["init_assigned_class"] = s_df["score_rank"].map(lambda x: x % 8).map(class_dic)
    init_flag = {(s, c): 0 for s in S for c in C}
    for row in s_df.itertuples():
        init_flag[row.student_id, row.init_assigned_class] = 1

    prob += pulp.lpSum([x[s, c] * init_flag[s, c] for s, c in SC])
    status = prob.solve()

    result = {
        "status": pulp.LpStatus[status],
        "objective": prob.objective.value(),
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
