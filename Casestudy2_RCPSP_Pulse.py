### Complex Scheduling SS2018
# Case Study 2: Group 1, Pulse Variables
# Resource-Constrained Project Scheduling Problem (RCPSP)

import gurobipy as grb
import numpy as np
import pandas as pd
import time

from collections import OrderedDict

grb.setParam('OutputFlag', 0)


class RCPSP_Solver():

    def __init__(self, xls, sheet_name):
        self._N = 32  # 30 + 2 dummy activities
        self._k = 4  # k different resources
        self._MAX_RUNTIME = 10 * 60  # 10 minutes

        self._sheet_name = sheet_name
        self._get_data(xls, sheet_name)

        self._model_DT = self._create_model(model_formulation='DT')
        self._model_DDT = self._create_model(model_formulation='DDT')

        self._pulse_vars = None

    def _get_data(self, xls, sheet_name):
        df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2, header=None)

        data = OrderedDict()
        df_listed = df.values.tolist()
        data['R'] = df_listed[0][:4]  # resource capacity, always k=4 different resources
        data['nodes'] = {}

        for row, node_id in zip(df_listed[1:], range(0, 31 + 1)):
            d = int(row[0])
            r = [int(i) for i in row[1:5]]
            s = int(row[5])
            succ = np.array(row[6:])
            succ = succ[~np.isnan(succ)]
            succ = [int(i) for i in succ]
            succ = [i-1 for i in succ]  # change to 0-indexed scheme

            assert (s == len(succ))

            data['nodes'][node_id] = {
                'd': d,
                'r': r,
                's': s,
                'succ': succ
            }
        self._data = data

    def _create_model(self, model_formulation):
        # create model
        model = grb.Model('RCPSP_' + self._sheet_name + '_' + model_formulation)

        T_MAX = sum([node_data['d'] for _, node_data in self._data['nodes'].items()])

        # add variables (pulse variables x_i,t in {0, 1} for i in V, t in H
        self._pulse_vars = []
        for i in range(0, self._N):
            self._pulse_vars.append([])
            for t in range(0, T_MAX + 1):
                self._pulse_vars[i].append(model.addVar(vtype=grb.GRB.BINARY,
                                                        name=('x_' + str(i) + '_' + str(t))))
        model.update()

        # set objective
        objective = grb.quicksum([t * x_last_t
                                  for t, x_last_t
                                  in zip(range(0, T_MAX + 1), self._pulse_vars[-1])])
        model.setObjective(objective, grb.GRB.MINIMIZE)

        # constraints:
        # (2.6): already declared as binary decision variable
        # (2.5): ES = 0, LS = T_MAX (naive implementation, no heuristic used)
        # (2.4): each variable has only one start time
        for i in range(0, self._N):
            model.addConstr(grb.quicksum(self._pulse_vars[i]) == 1, name='c_2.4_' + str(i))

        # (2.3): resource constraints
        for k, R_k, in enumerate(self._data['R']):
            for t in range(0, T_MAX + 1):
                model.addConstr(
                    grb.quicksum(
                        sum(
                            [[self._data['nodes'][i]['r'][k] * self._pulse_vars[i][tau]
                              for tau in range(t - self._data['nodes'][i]['d'] + 1, t+1)] for i in range(0, self._N)]
                            , [])  # trick to flatten list
                    ) <= R_k,
                    'c_2.3_' + str(k) + '_' + str(t))

            # (2.2) / (2.7)
            if model_formulation == 'DT':
                # (2.2): precedence constraints DT
                for i in range(0, self._N):
                    for j in self._data['nodes'][i]['succ']:
                        model.addConstr(sum([t * self._pulse_vars[j][t] for t in range(0, T_MAX + 1)]) -
                                        sum([t * self._pulse_vars[i][t] for t in range(0, T_MAX + 1)]) >=
                                        self._data['nodes'][i]['d'],
                                        'c_2.2_' + str(i) + '_' + str(j))
            if model_formulation == 'DDT':
                # (2.7): precedence constraints DDT
                for i in range(0, self._N):
                    for j in self._data['nodes'][i]['succ']:
                        for t in range(0, T_MAX + 1):
                            model.addConstr(sum([self._pulse_vars[i][tau] for tau in range(0, t - self._data['nodes'][i]['d'])])
                                            - sum([self._pulse_vars[j][tau] for tau in range(0, t)]) >= 0,
                                            'c_2.7_' + str(i) + '_' + str(j) + '_' + str(t))

        model.update()

        return model

    def run(self, model_formulation):
        """Runs either the 'DT' or the 'DDT' formulation of the problem."""

        if model_formulation == 'DT':
            model = self._model_DT
        elif model_formulation == 'DDT':
            model = self._model_DDT
        else:
            raise ValueError('No "', model_formulation, '" as `model_formulation` found!')

        model.setParam('TimeLimit', self._MAX_RUNTIME)

        tick = time.clock()
        model.optimize()
        tock = time.clock()

        runtime = tock - tick

        print('case:', self._sheet_name)
        print('formulation:', model_formulation)
        print('Objective Value:', model.objVal)
        print('Runtime:', runtime)
        print('---------')

        return model.objVal, runtime


if __name__ == '__main__':

    data_path = 'data/RCPSP_data.xls'
    output_path = 'CS2018_casestudy2_results.xlsx'

    xls = pd.ExcelFile(data_path)
    xls_names = xls.sheet_names[1:]  # skip first sheet 'results'

    opt_obj_values_DT = []
    runtimes_DT = []
    opt_obj_values_DDT = []
    runtimes_DDT = []

    for sheet_name in xls_names:
        solver = RCPSP_Solver(xls, sheet_name)

        value_DT, rt_DT = solver.run('DT')
        opt_obj_values_DT.append(value_DT)
        runtimes_DT.append(rt_DT)

        value_DDT, rt_DDT = solver.run('DDT')
        opt_obj_values_DDT.append(value_DDT)
        runtimes_DDT.append(rt_DDT)

    results = pd.DataFrame({'case': xls_names,
                            'DT_objective_value': opt_obj_values_DT, 'DT_runtime_in_s': runtimes_DT,
                            'DDT_objective_value': opt_obj_values_DDT, 'DDT_runtime_in_s': runtimes_DDT})

    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    results.to_excel(writer, sheet_name='results')
    writer.save()
