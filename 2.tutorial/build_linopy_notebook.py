"""tutorial.ipynb を元に tutorial_linopy.ipynb を生成する（linopy版・検証付き）"""
import json

with open("tutorial.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

# linopy 用のコードに差し替え（セルインデックス -> source 文字列）
linopy_sources = {
    3: """from linopy import Model

m = Model()
x = m.add_variables(name='x')
y = m.add_variables(name='y')

m.add_constraints(120 * x + 150 * y == 1440)
m.add_constraints(x + y == 10)
m.add_objective(0 * x)  # 連立方程式なので目的は0（linopyは式のみ受け付ける）

status, cond = m.solve(solver_name='highs')

print('Status:', status)
print('x=', float(x.solution), 'y=', float(y.solution))""",
    5: """# Pythonライブラリlinopyの取り込み
from linopy import Model""",
    6: """# 数理モデルの定義
m = Model()
m""",
    7: """# 変数の定義
x = m.add_variables(name='x')
y = m.add_variables(name='y')""",
    8: """# 制約式の定義
m.add_constraints(120 * x + 150 * y == 1440)
m.add_constraints(x + y == 10)
m""",
    9: """# 目的関数（0）を設定して求解
m.add_objective(0 * x)
status, cond = m.solve(solver_name='highs')
print('Status:', status)""",
    10: """# 最適化結果の表示
print('x=', float(x.solution), 'y=', float(y.solution))""",
    13: """from linopy import Model

m = Model()
x = m.add_variables(lower=0, name='x')
y = m.add_variables(lower=0, name='y')

m.add_constraints(1 * x + 3 * y <= 30)
m.add_constraints(2 * x + 1 * y <= 40)
m.add_constraints(x >= 0)
m.add_constraints(y >= 0)
m.add_objective(x + 2 * y, sense='max')

status, cond = m.solve(solver_name='highs')

print('Status:', status)
print('x=', float(x.solution), 'y=', float(y.solution), 'obj=', m.objective.value)""",
    15: """# Pythonライブラリlinopyの取り込み
from linopy import Model""",
    16: """# 数理最適化モデルの定義
m = Model()
m""",
    17: """# 変数の定義
x = m.add_variables(lower=0, name='x')
y = m.add_variables(lower=0, name='y')""",
    18: """# 制約式の定義
m.add_constraints(1 * x + 3 * y <= 30)
m.add_constraints(2 * x + 1 * y <= 40)
m.add_constraints(x >= 0)
m.add_constraints(y >= 0)""",
    19: """# 目的関数の定義
m.add_objective(x + 2 * y, sense='max')
m""",
    20: """# 求解
status, cond = m.solve(solver_name='highs')
print('Status:', status)""",
    21: """# 最適化結果の表示
print('x=', float(x.solution), 'y=', float(y.solution), 'obj=', m.objective.value)""",
    25: """# データ処理のためのライブラリpandasとPythonライブラリlinopyの取り込み
import pandas as pd
from linopy import Model""",
    37: """# 数理最適化モデルの定義
model = Model()""",
    39: """# 変数の定義（linopyでは次元付き変数として1つで定義）
import xarray as xr
coords_p = xr.DataArray(P, dims=['p'])
x = model.add_variables(lower=0, coords=[coords_p], name='x')""",
    41: """# 生産量は0以上（PuLP と同様に明示制約として追加し ncons=7 に揃える）
model.add_constraints(x >= 0)""",
    42: """# 生産量は在庫の範囲
require_arr = xr.DataArray([[require[p, m] for m in M] for p in P], coords=[P, M], dims=['p', 'm'])
model.add_constraints((x * require_arr).sum('p') <= xr.DataArray([stock[mat] for mat in M], coords=[M], dims=['m']))""",
    44: """# 目的関数の定義
gain_arr = xr.DataArray([gain[p] for p in P], coords=[P], dims=['p'])
model.add_objective((x * gain_arr).sum(), sense='max')""",
    46: """# 求解
status, cond = model.solve(solver_name='highs')
print('Status:', status)""",
    47: """# 計算結果の表示
for p in P:
    print(p, float(x.solution.sel(p=p)))

print('obj=', model.objective.value)""",
    49: """import pandas as pd
import xarray as xr
from linopy import Model

# データの取得
require_df = pd.read_csv('requires.csv')
stock_df = pd.read_csv('stocks.csv')
gain_df = pd.read_csv('gains.csv')

# 集合の定義
P = gain_df['p'].tolist()
M = stock_df['m'].tolist()

# 定数の定義
stock = {row.m:row.stock for row in stock_df.itertuples()}
gain = {row.p:row.gain for row in gain_df.itertuples()}
require = {(row.p,row.m):row.require for row in require_df.itertuples()}

# 数理最適化モデルの定義（線形計画）
model = Model()
coords_p = xr.DataArray(P, dims=['p'])
x = model.add_variables(lower=0, coords=[coords_p], name='x')
require_arr = xr.DataArray([[require[p, m] for m in M] for p in P], coords=[P, M], dims=['p', 'm'])
model.add_constraints((x * require_arr).sum('p') <= xr.DataArray([stock[mat] for mat in M], coords=[M], dims=['m']))
gain_arr = xr.DataArray([gain[p] for p in P], coords=[P], dims=['p'])
model.add_constraints(x >= 0)
model.add_objective((x * gain_arr).sum(), sense='max')

# 求解
status, cond = model.solve(solver_name='highs')
print('Status:', status)

# 計算結果の表示
for p in P:
    print(p, float(x.solution.sel(p=p)))

print('obj=', model.objective.value)""",
    52: """import pandas as pd
import xarray as xr
from linopy import Model

# データの取得
require_df = pd.read_csv('requires.csv')
stock_df = pd.read_csv('stocks.csv')
gain_df = pd.read_csv('gains.csv')

# 集合の定義
P = gain_df['p'].tolist()
M = stock_df['m'].tolist()

# 定数の定義
stock = {row.m:row.stock for row in stock_df.itertuples()}
gain = {row.p:row.gain for row in gain_df.itertuples()}
require = {(row.p,row.m):row.require for row in require_df.itertuples()}

# 数理最適化モデルの定義（整数計画）
model = Model()
coords_p = xr.DataArray(P, dims=['p'])
x = model.add_variables(lower=0, coords=[coords_p], name='x', integer=True)
require_arr = xr.DataArray([[require[p, m] for m in M] for p in P], coords=[P, M], dims=['p', 'm'])
model.add_constraints(x >= 0)
model.add_constraints((x * require_arr).sum('p') <= xr.DataArray([stock[mat] for mat in M], coords=[M], dims=['m']))
gain_arr = xr.DataArray([gain[p] for p in P], coords=[P], dims=['p'])
model.add_objective((x * gain_arr).sum(), sense='max')

# 求解（整数計画は highs が対応）
status, cond = model.solve(solver_name='highs')
print('Status:', status)

# 計算結果の表示
for p in P:
    print(p, float(x.solution.sel(p=p)))

print('obj=', model.objective.value)""",
}


def to_source_lines(s):
    return [line + "\n" for line in s.split("\n")]


# 新しいノートブック: 同じセル構造で、コードセルを linopy に差し替え、出力は空に
cells = []
for i, cell in enumerate(nb["cells"]):
    new_cell = {"cell_type": cell["cell_type"], "metadata": cell.get("metadata", {})}
    if cell["cell_type"] == "code":
        src = linopy_sources.get(i, cell["source"])
        if isinstance(src, str):
            new_cell["source"] = to_source_lines(src)
        else:
            new_cell["source"] = cell["source"]
        new_cell["outputs"] = []
        new_cell["execution_count"] = None
    else:
        new_cell["source"] = cell["source"]
    cells.append(new_cell)

# 検証セルを追加
verify_code = """import json
import os

MIP_GAP = 1e-6
expected_path = os.path.join(os.path.dirname(os.path.abspath('')), 'baseline_expected.json')
with open(expected_path, encoding='utf-8') as f:
    expected = json.load(f)

def check(name, obj, nvars, ncons):
    e = expected[name]
    ok_obj = abs(obj - e['objective']) <= MIP_GAP or (e['objective'] != 0 and abs((obj - e['objective']) / e['objective']) <= MIP_GAP)
    assert ok_obj, f"{name}: objective {obj} != expected {e['objective']}"
    assert nvars == e['nvars'], f"{name}: nvars {nvars} != expected {e['nvars']}"
    assert ncons == e['ncons'], f"{name}: ncons {ncons} != expected {e['ncons']}"
    print(f'{name}: OK (obj={obj}, nvars={nvars}, ncons={ncons})')

# SLE, LP, LP2, IP を再実行して検証
m_sle = Model(); x_sle = m_sle.add_variables(name='x'); y_sle = m_sle.add_variables(name='y')
m_sle.add_constraints(120*x_sle + 150*y_sle == 1440); m_sle.add_constraints(x_sle + y_sle == 10)
m_sle.add_objective(0 * x_sle); m_sle.solve(solver_name='highs')
check('SLE', m_sle.objective.value, m_sle.variables.nvars, m_sle.constraints.ncons)

m_lp = Model(); x_lp = m_lp.add_variables(lower=0, name='x'); y_lp = m_lp.add_variables(lower=0, name='y')
m_lp.add_constraints(1*x_lp + 3*y_lp <= 30); m_lp.add_constraints(2*x_lp + 1*y_lp <= 40)
m_lp.add_constraints(x_lp >= 0); m_lp.add_constraints(y_lp >= 0)
m_lp.add_objective(x_lp + 2*y_lp, sense='max'); m_lp.solve(solver_name='highs')
check('LP', m_lp.objective.value, m_lp.variables.nvars, m_lp.constraints.ncons)

model_lp2 = Model(); x_lp2 = model_lp2.add_variables(lower=0, coords=[xr.DataArray(P, dims=['p'])], name='x')
model_lp2.add_constraints(x_lp2 >= 0)
model_lp2.add_constraints((x_lp2 * require_arr).sum('p') <= xr.DataArray([stock[mat] for mat in M], coords=[M], dims=['m']))
model_lp2.add_objective((x_lp2 * gain_arr).sum(), sense='max'); model_lp2.solve(solver_name='highs')
check('LP2', model_lp2.objective.value, model_lp2.variables.nvars, model_lp2.constraints.ncons)

model_ip = Model(); x_ip = model_ip.add_variables(lower=0, coords=[xr.DataArray(P, dims=['p'])], name='x', integer=True)
model_ip.add_constraints(x_ip >= 0)
model_ip.add_constraints((x_ip * require_arr).sum('p') <= xr.DataArray([stock[mat] for mat in M], coords=[M], dims=['m']))
model_ip.add_objective((x_ip * gain_arr).sum(), sense='max'); model_ip.solve(solver_name='highs')
check('IP', model_ip.objective.value, model_ip.variables.nvars, model_ip.constraints.ncons)

print('\\nすべての検証に合格しました。')"""

cells.append({"cell_type": "markdown", "metadata": {}, "source": ["### 検証：最適値・変数数・制約数がベースラインと一致することの確認\n"]})
cells.append({"cell_type": "code", "metadata": {}, "source": to_source_lines(verify_code), "outputs": [], "execution_count": None})

out = {
    "cells": cells,
    "metadata": nb["metadata"],
    "nbformat": nb["nbformat"],
    "nbformat_minor": nb["nbformat_minor"],
}
with open("tutorial_linopy.ipynb", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("Created tutorial_linopy.ipynb")
