"""第6章 API 車グループ分け: PuLP の実行結果を取得（検証用ベースライン）"""
import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import pandas as pd
    import pulp
    from problem import CarGroupProblem
except ImportError as e:
    print("Required: pandas, pulp (problem.py)", file=sys.stderr)
    sys.exit(1)


def main():
    students_df = pd.read_csv("resource/students.csv")
    cars_df = pd.read_csv("resource/cars.csv")
    prob = CarGroupProblem(students_df, cars_df)
    prob.solve()
    pulp_prob = prob.prob["prob"]
    obj_val = 0.0
    if pulp_prob.objective is not None and pulp_prob.objective.value() is not None:
        obj_val = pulp_prob.objective.value()
    # 意思決定変数 x の数（PuLP は補助変数で 145 になることがあるため）
    n_x = len(prob.prob["list"]["S"]) * len(prob.prob["list"]["C"])
    result = {
        "status": pulp_prob.status,
        "objective": obj_val,
        "nvars": n_x,
        "ncons": len(pulp_prob.constraints),
    }
    out_path = "baseline_pulp.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Saved", out_path)
    print(result)
    return result


if __name__ == "__main__":
    main()
