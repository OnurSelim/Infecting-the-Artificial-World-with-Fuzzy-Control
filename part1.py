from plague import Plague
from skfuzzy import control as ctrl
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

iteration_no = 200
count_th = 20

plague = Plague()

infection = ctrl.Antecedent(np.arange(0, 1, 0.001), 'infection')
cont_var = ctrl.Consequent(np.arange(-0.15, 0.15, 0.001), 'control_var')

# Test Case 1
# infection['low'] = fuzz.trimf(infection.universe, [0, 0, 0.52])
# infection['medium'] = fuzz.trimf(infection.universe, [0.4, 0.6, 0.7])
# infection['high'] = fuzz.trimf(infection.universe, [0.60, 1, 1])

# cont_var['negative'] = fuzz.trimf(cont_var.universe, [-0.15, -0.15, 0])
# cont_var['medium'] = fuzz.trimf(cont_var.universe, [-0.15, 0, 0.15])
# cont_var['positive'] = fuzz.trimf(cont_var.universe, [0, 0.15, 0.15])

# Test Case 2
# infection['low'] = fuzz.gaussmf(infection.universe, 0, 0.20)
# infection['medium'] = fuzz.gaussmf(infection.universe, 0.6, 0.20)
# infection['high'] = fuzz.gaussmf(infection.universe, 1, 0.116)
#
# cont_var['negative'] = fuzz.trimf(cont_var.universe, [-0.15, -0.15, 0])
# cont_var['medium'] = fuzz.trimf(cont_var.universe, [-0.15, 0, 0.15])
# cont_var['positive'] = fuzz.trimf(cont_var.universe, [0, 0.15, 0.15])

# Test Case 3
infection['low'] = fuzz.trapmf(infection.universe, [0, 0, 0.59, 0.6])
infection['medium'] = fuzz.trimf(infection.universe, [0.59, 0.6, 0.61])
infection['high'] = fuzz.trapmf(infection.universe, [0.6, 0.61, 1, 1])
#
cont_var['negative'] = fuzz.gaussmf(cont_var.universe, -0.15, 0.02)
cont_var['medium'] = fuzz.gaussmf(cont_var.universe, 0, 0.02)
cont_var['positive'] = fuzz.gaussmf(cont_var.universe, 0.15, 0.02)

infection.view()
cont_var.view()

rule1 = ctrl.Rule(infection['low'], cont_var['positive'])
rule2 = ctrl.Rule(infection['medium'], cont_var['medium'])
rule3 = ctrl.Rule(infection['high'], cont_var['negative'])

ctrl_rule = ctrl.ControlSystem([rule1, rule2, rule3])
var_controlling = ctrl.ControlSystemSimulation(ctrl_rule)
for k in range(iteration_no):
    (curr_inf, cur_inf_rate) = plague.checkInfectionStatus()
    var_controlling.input['infection'] = curr_inf
    var_controlling.compute()

    curr_cont_var = var_controlling.output['control_var']
    print(k, curr_inf, var_controlling.output['control_var'])
    # cont_var.view(sim=var_controlling)
    plague.spreadPlague(curr_cont_var)
    # cont_var.view(sim=var_controlling)

arr_infected = np.asarray(plague.infected_percentage_curve_)

counter = 0
ss = 0

for k in range(len(arr_infected)):
    if abs(arr_infected[k] - 0.6) > 0.001:
        counter = 0
    elif abs(arr_infected[k] - 0.6) < 0.001 and counter == 0:
        ss = k
        counter = counter + 1
    elif abs(arr_infected[k] - 0.6) < 0.001 and counter == count_th:
        break
    elif abs(arr_infected[k] - 0.6) < 0.001:
        counter = counter + 1

arr_infection_rate = np.asarray(plague.infection_rate_curve_)
cost = np.sum(arr_infection_rate[0:ss])

plague.viewPlague(ss, 10*cost, show_plot=True)
