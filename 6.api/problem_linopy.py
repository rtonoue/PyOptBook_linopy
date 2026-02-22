"""学生の乗車グループ分け問題を linopy で解く版（problem.py と同じインターフェース）"""
import pandas as pd
import xarray as xr
from linopy import Model


class CarGroupProblem():
    """学生の乗車グループ分け問題を解くクラス（linopy 実装）"""
    def __init__(self, students_df, cars_df, name='ClubCarProblem'):
        self.students_df = students_df
        self.cars_df = cars_df
        self.name = name
        self.prob = self._formulate()

    def _formulate(self):
        S = self.students_df['student_id'].to_list()
        C = self.cars_df['car_id'].to_list()
        G = [1, 2, 3, 4]
        S_license = self.students_df[self.students_df['license'] == 1]['student_id'].tolist()
        S_g = {g: self.students_df[self.students_df['grade'] == g]['student_id'].tolist() for g in G}
        S_male = self.students_df[self.students_df['gender'] == 0]['student_id'].tolist()
        S_female = self.students_df[self.students_df['gender'] == 1]['student_id'].tolist()
        U = self.cars_df.set_index('car_id')['capacity'].to_dict()

        S_arr = xr.DataArray(S, dims=['s'])
        C_arr = xr.DataArray(C, dims=['c'])
        model = Model()
        x = model.add_variables(coords=[S_arr, C_arr], name='x', binary=True)

        # (1) 各学生を1つの車に割り当てる
        model.add_constraints(x.sum('c') == 1)

        # (2) 各車の乗車定員
        for c in C:
            cap = U[c]
            model.add_constraints(x.sel(c=c).sum('s') <= cap)

        # (3) 各車にドライバー1人以上
        for c in C:
            mask = xr.DataArray([1 if s in S_license else 0 for s in S], coords=[S_arr], dims=['s'])
            model.add_constraints((x.sel(c=c) * mask).sum('s') >= 1)

        # (4) 各車に各学年1人以上
        for c in C:
            for g in G:
                mask = xr.DataArray([1 if s in S_g[g] else 0 for s in S], coords=[S_arr], dims=['s'])
                model.add_constraints((x.sel(c=c) * mask).sum('s') >= 1)

        # (5)(6) 各車に男性・女性1人以上
        for c in C:
            mask_m = xr.DataArray([1 if s in S_male else 0 for s in S], coords=[S_arr], dims=['s'])
            model.add_constraints((x.sel(c=c) * mask_m).sum('s') >= 1)
        for c in C:
            mask_f = xr.DataArray([1 if s in S_female else 0 for s in S], coords=[S_arr], dims=['s'])
            model.add_constraints((x.sel(c=c) * mask_f).sum('s') >= 1)

        # 目的は定数（ feasibility ）なので 0*x の和
        model.add_objective((0 * x).sum())
        return {'model': model, 'variable': {'x': x}, 'list': {'S': S, 'C': C}}

    def solve(self):
        self.prob['model'].solve(solver_name='highs')
        x = self.prob['variable']['x']
        S = self.prob['list']['S']
        C = self.prob['list']['C']
        xvals = x.solution.values  # shape (len(S), len(C))
        car2students = {C[j]: [S[i] for i in range(len(S)) if xvals[i, j] >= 0.5] for j in range(len(C))}
        student2car = {s: c for c, ss in car2students.items() for s in ss}
        solution_df = pd.DataFrame(list(student2car.items()), columns=['student_id', 'car_id'])
        return solution_df


if __name__ == '__main__':
    students_df = pd.read_csv('resource/students.csv')
    cars_df = pd.read_csv('resource/cars.csv')
    prob = CarGroupProblem(students_df, cars_df)
    solution_df = prob.solve()
    print('Solution:\n', solution_df)
