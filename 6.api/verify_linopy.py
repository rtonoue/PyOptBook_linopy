"""第6章 API: linopy 版（problem_linopy.py）の結果が PuLP ベースラインと一致するか検証"""
import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from problem_linopy import CarGroupProblem

MIP_GAP = 1e-5
with open("baseline_pulp.json", encoding="utf-8") as f:
    expected = json.load(f)

students_df = pd.read_csv("resource/students.csv")
cars_df = pd.read_csv("resource/cars.csv")
prob = CarGroupProblem(students_df, cars_df)
prob.solve()
model = prob.prob["model"]
obj = model.objective.value
nvars = model.variables.nvars
ncons = model.constraints.ncons

assert abs(obj - expected["objective"]) <= MIP_GAP, f"objective {obj} != {expected['objective']}"
assert nvars == expected["nvars"], f"nvars {nvars} != {expected['nvars']}"
assert ncons == expected["ncons"], f"ncons {ncons} != {expected['ncons']}"
print("API CarGroupProblem: OK", "obj=", obj, "nvars=", nvars, "ncons=", ncons)
print("検証に合格しました。")
