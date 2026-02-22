"""第3章 学校のクラス編成: linopy の計算結果が PuLP ベースラインと一致するか検証"""
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import xarray as xr
from linopy import Model

MIP_GAP = 1e-5
with open("baseline_pulp.json", encoding="utf-8") as f:
    expected = json.load(f)


def check(obj, nvars, ncons):
    e = expected
    assert abs(obj - e["objective"]) <= MIP_GAP, f"objective {obj} != {e['objective']}"
    assert nvars == e["nvars"], f"nvars {nvars} != {e['nvars']}"
    assert ncons == e["ncons"], f"ncons {ncons} != {e['ncons']}"
    print("School: OK", "obj=", obj, "nvars=", nvars, "ncons=", ncons)


s_df = pd.read_csv("students.csv")
s_pair_df = pd.read_csv("student_pairs.csv")
S = s_df["student_id"].tolist()
C = ["A", "B", "C", "D", "E", "F", "G", "H"]
SC = [(s, c) for s in S for c in C]

S_arr = xr.DataArray(S, dims=["s"])
C_arr = xr.DataArray(C, dims=["c"])
model = Model()
x = model.add_variables(coords=[S_arr, C_arr], name="x", binary=True)

# (1) 各生徒は1つのクラスに割り当てる
model.add_constraints(x.sum("c") == 1)
# (2) 各クラスの人数 39〜40
model.add_constraints(x.sum("s") >= 39)
model.add_constraints(x.sum("s") <= 40)

S_male = [row.student_id for row in s_df.itertuples() if row.gender == 1]
S_female = [row.student_id for row in s_df.itertuples() if row.gender == 0]
mask_male = xr.DataArray([1 if s in S_male else 0 for s in S], coords=[S_arr], dims=["s"])
mask_female = xr.DataArray([1 if s in S_female else 0 for s in S], coords=[S_arr], dims=["s"])
model.add_constraints((x * mask_male).sum("s") <= 20)
model.add_constraints((x * mask_female).sum("s") <= 20)

score = {row.student_id: row.score for row in s_df.itertuples()}
score_arr = xr.DataArray([score[s] for s in S], coords=[S_arr], dims=["s"])
score_mean = float(s_df["score"].mean())
model.add_constraints((x * score_arr).sum("s") >= (score_mean - 10) * x.sum("s"))
model.add_constraints((x * score_arr).sum("s") <= (score_mean + 10) * x.sum("s"))

S_leader = [row.student_id for row in s_df.itertuples() if row.leader_flag == 1]
mask_leader = xr.DataArray([1 if s in S_leader else 0 for s in S], coords=[S_arr], dims=["s"])
model.add_constraints((x * mask_leader).sum("s") >= 2)

S_support = [row.student_id for row in s_df.itertuples() if row.support_flag == 1]
mask_support = xr.DataArray([1 if s in S_support else 0 for s in S], coords=[S_arr], dims=["s"])
model.add_constraints((x * mask_support).sum("s") <= 1)

SS = [(row.student_id1, row.student_id2) for row in s_pair_df.itertuples()]
for s1, s2 in SS:
    model.add_constraints(x.sel(s=s1) + x.sel(s=s2) <= 1)

s_df = s_df.copy()
s_df["score_rank"] = s_df["score"].rank(ascending=False, method="first")
class_dic = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
s_df["init_assigned_class"] = s_df["score_rank"].map(lambda x: x % 8).map(class_dic)
init_flag = {(s, c): 0 for s in S for c in C}
for row in s_df.itertuples():
    init_flag[row.student_id, row.init_assigned_class] = 1
init_flag_arr = xr.DataArray(
    [[init_flag[s, c] for c in C] for s in S], coords=[S_arr, C_arr], dims=["s", "c"]
)
model.add_objective((x * init_flag_arr).sum(), sense="max")
model.solve(solver_name="highs")

check(model.objective.value, model.variables.nvars, model.constraints.ncons)
print("検証に合格しました。")
