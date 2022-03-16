from plague import Plague
from skfuzzy import control as ctrl
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

iteration_no = 200
count_th = 75

plague = Plague()

infection = ctrl.Antecedent(np.arange(0, 1, 0.001), 'infection')
infection_rate = ctrl.Antecedent(np.arange(-1, 1, 0.001), 'infection_rate')
cont_var = ctrl.Consequent(np.arange(-0.15, 0.15, 0.001), 'control_var')

# Test Case 1
# infection['low'] = fuzz.gaussmf(infection.universe, 0, 0.178)
# infection['medium'] = fuzz.gaussmf(infection.universe, 0.6, 0.10)
# infection['high'] = fuzz.gaussmf(infection.universe, 1, 0.12)
#
# infection_rate['low'] = fuzz.trimf(infection_rate.universe, [-1, -1, 0])
# infection_rate['medium'] = fuzz.trimf(infection_rate.universe, [-1, 0, 1])
# infection_rate['high'] = fuzz.trimf(infection_rate.universe, [0, 1, 1])
#
# cont_var['nn'] = fuzz.gaussmf(cont_var.universe, -0.15, 0.015)
# cont_var['nm'] = fuzz.gaussmf(cont_var.universe, -0.15/2, 0.015)
# cont_var['mm'] = fuzz.gaussmf(cont_var.universe, 0, 0.015)
# cont_var['pm'] = fuzz.gaussmf(cont_var.universe, 0.15/2, 0.015)
# cont_var['pp'] = fuzz.gaussmf(cont_var.universe, 0.15, 0.015)

# Test Case 2
infection['low'] = fuzz.trimf(infection.universe, [0, 0, 0.6])
infection['medium'] = fuzz.trimf(infection.universe, [0.5, 0.6, 0.7])
infection['high'] = fuzz.trimf(infection.universe, [0.6, 1, 1])
#
infection_rate['low'] = fuzz.trimf(infection_rate.universe, [-1, -1, 0])
infection_rate['medium'] = fuzz.trimf(infection_rate.universe, [-0.5, 0, 0.5])
infection_rate['high'] = fuzz.trimf(infection_rate.universe, [0, 1, 1])
#
cont_var['nn'] = fuzz.gaussmf(cont_var.universe, -0.15, 0.02)
cont_var['nm'] = fuzz.gaussmf(cont_var.universe, -0.15/2, 0.002)
cont_var['mm'] = fuzz.gaussmf(cont_var.universe, 0, 0.002)
cont_var['pm'] = fuzz.gaussmf(cont_var.universe, 0.15/2, 0.002)
cont_var['pp'] = fuzz.gaussmf(cont_var.universe, 0.15, 0.02)

infection.view()
infection_rate.view()
cont_var.view()

# Rule Set 1
rule1 = ctrl.Rule(infection['low'] | infection_rate['low'], cont_var['pp'])
rule2 = ctrl.Rule(infection['low'] | infection_rate['medium'], cont_var['pm'])
rule3 = ctrl.Rule(infection['low'] | infection_rate['high'], cont_var['mm'])
rule4 = ctrl.Rule(infection['medium'] | infection_rate['low'], cont_var['pm'])
rule5 = ctrl.Rule(infection['medium'] | infection_rate['medium'], cont_var['mm'])
rule6 = ctrl.Rule(infection['medium'] | infection_rate['high'], cont_var['nm'])
rule7 = ctrl.Rule(infection['high'] | infection_rate['low'], cont_var['mm'])
rule8 = ctrl.Rule(infection['high'] | infection_rate['medium'], cont_var['nm'])
rule9 = ctrl.Rule(infection['high'] | infection_rate['high'], cont_var['nn'])

# Rule Set 2
# rule1 = ctrl.Rule(infection['low'] & infection_rate['low'], cont_var['pp'])
# rule2 = ctrl.Rule(infection['low'] & infection_rate['medium'], cont_var['pm'])
# rule3 = ctrl.Rule(infection['low'] & infection_rate['high'], cont_var['mm'])
# rule4 = ctrl.Rule(infection['medium'] & infection_rate['low'], cont_var['pm'])
# rule5 = ctrl.Rule(infection['medium'] & infection_rate['medium'], cont_var['mm'])
# rule6 = ctrl.Rule(infection['medium'] & infection_rate['high'], cont_var['nm'])
# rule7 = ctrl.Rule(infection['high'] & infection_rate['low'], cont_var['mm'])
# rule8 = ctrl.Rule(infection['high'] & infection_rate['medium'], cont_var['nm'])
# rule9 = ctrl.Rule(infection['high'] & infection_rate['high'], cont_var['nn'])

ctrl_rule = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
var_controlling = ctrl.ControlSystemSimulation(ctrl_rule)
for k in range(iteration_no):
    (curr_inf, curr_inf_rate) = plague.checkInfectionStatus()
    var_controlling.input['infection'] = curr_inf
    var_controlling.input['infection_rate'] = curr_inf_rate
    var_controlling.compute()

    curr_cont_var = var_controlling.output['control_var']
    print(k, curr_inf, curr_inf_rate, var_controlling.output['control_var'])
    cont_var.view(sim=var_controlling)
    # plague.spreadPlague(curr_cont_var)

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

