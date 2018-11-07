### Complex Scheduling SS2018
# Case Study 1
# Generic Time Scheduling Problem

import gurobipy as grb
import pandas as pd
import time

grb.setParam('OutputFlag', 0)

class GTSP_Solver():

    def __init__(self, N, T_MAX, data_path, sheet_name):
        self.N = N
        self.T_MAX = T_MAX

        self._sheet_name = sheet_name
        self._get_data(data_path, sheet_name)

        self._model = self._create_model()
        self._starting_times = None
        self._objective = None

    def _get_data(self, file_path, sheet_name):
        self._data = pd.read_excel(file_path, skiprows=2, sheet_name=sheet_name)
        self._data['i'] = self._data['i'].astype(int)
        self._data['j'] = self._data['j'].astype(int)
        self._data['delta_ij'] = self._data['delta_ij'].astype(int)

    def _create_model(self):
        # create model
        model = grb.Model('GTSP_' + str(self.N) + '_' + str(self.T_MAX))

        # add variables (starting times)
        self._starting_times = []
        for i in range(self.N + 1):
                self._starting_times.append(model.addVar(lb=0, ub=self.T_MAX,
                                                             vtype=grb.GRB.INTEGER, name=('S_' + str(i))))

        # add end dummy node: set upper bound to T_MAX
        self._starting_times.append(model.addVar(lb=0, vtype=grb.GRB.INTEGER, name=('S_' + str(self.N + 2))))

        model.update()

        # set objective
        objective = grb.quicksum(self._starting_times)
        model.setObjective(objective, grb.GRB.MINIMIZE)

        # constraints: S_i >= 0 and S_n+1 <= T_max already set in variable initialization
        # add constraints: S_j - S_i >= delta_ij
        for _, row in self._data.iterrows():
            i = int(row['i'])
            j = int(row['j'])
            delta_ij = int(row['delta_ij'])

            model.addConstr(self._starting_times[j] - self._starting_times[i] >=
                                  delta_ij, name=('c_S_' + str(j) + '_S_' + str(i)))

        # add constraint: S_0 = 0
        model.addConstr(self._starting_times[0] == 0, 'c_S0_eq_0')

        model.update()

        return model

    def run_GTSP(self):

        tick = time.clock()
        self._model.optimize()
        tock = time.clock()

        runtime = tock - tick

        print('n:', self.N, 'T_max:', self.T_MAX)
        print('Objective Value:', self._model.objVal)
        print('Runtime:', runtime)
        print('---------')

        return self._model.objVal, runtime


if __name__ == '__main__':

    data_path = 'data/PSP_Daten.xls'
    output_path = 'CS2018_casestudy1_results.xlsx'

    N = (10, 100, 1000)
    T_MAX = (20, 200, 2000)
    sheet_number = (0, 1, 2)

    opt_obj_values = []
    runtimes = []

    for n, t_max, sheet_nr in zip(N, T_MAX, sheet_number):
        solver = GTSP_Solver(n, t_max, data_path, sheet_nr)
        value, rt = solver.run_GTSP()
        opt_obj_values.append(value)
        runtimes.append(rt)

    results = pd.DataFrame({'n': N, 'T_max': T_MAX, 'objective_value': opt_obj_values, 'runtime_in_s': runtimes})

    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    results.to_excel(writer, sheet_name='results')
    writer.save()
