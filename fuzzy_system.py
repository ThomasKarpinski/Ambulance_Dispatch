import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

"""
Fuzzy Inference System for Emergency Dispatch Priority

Inputs:
- severity (reported priority): 1–5
- travel_time: minutes (0–60)

Output:
- priority score: 0–100
"""

# =========================
# FUZZY VARIABLES
# =========================

# Input: Severity (reported)
severity = ctrl.Antecedent(np.arange(0, 6.1, 0.1), 'severity')

# Input: Travel Time (minutes)
travel_time = ctrl.Antecedent(np.arange(0, 61, 1), 'travel_time')

# Output: Priority Score
priority = ctrl.Consequent(np.arange(0, 101, 1), 'priority')


# =========================
# MEMBERSHIP FUNCTIONS
# =========================

# Severity
severity['minor'] = fuzz.trapmf(severity.universe, [0, 0, 2, 3])
severity['moderate'] = fuzz.trimf(severity.universe, [2, 3, 4])
severity['critical'] = fuzz.trapmf(severity.universe, [3, 4, 6, 6])

# Travel time
travel_time['short'] = fuzz.trapmf(travel_time.universe, [0, 0, 10, 20])
travel_time['medium'] = fuzz.trimf(travel_time.universe, [10, 30, 50])
travel_time['long'] = fuzz.trapmf(travel_time.universe, [40, 50, 60, 60])

# Priority
priority['low'] = fuzz.trimf(priority.universe, [0, 0, 20])
priority['standard'] = fuzz.trimf(priority.universe, [20, 40, 60])
priority['urgent'] = fuzz.trimf(priority.universe, [60, 80, 90])
priority['life_threatening'] = fuzz.trimf(priority.universe, [90, 100, 100])

# =========================
# FUZZY RULES
# =========================

rules = [
    ctrl.Rule(severity['critical'] & travel_time['short'], priority['life_threatening']),
    ctrl.Rule(severity['critical'] & travel_time['medium'], priority['life_threatening']),
    ctrl.Rule(severity['critical'] & travel_time['long'], priority['urgent']),

    ctrl.Rule(severity['moderate'] & travel_time['short'], priority['urgent']),
    ctrl.Rule(severity['moderate'] & travel_time['medium'], priority['standard']),
    ctrl.Rule(severity['moderate'] & travel_time['long'], priority['low']),

    ctrl.Rule(severity['minor'] & travel_time['short'], priority['standard']),
    ctrl.Rule(severity['minor'] & travel_time['medium'], priority['low']),
    ctrl.Rule(severity['minor'] & travel_time['long'], priority['low']),
]

dispatch_ctrl = ctrl.ControlSystem(rules)


# =========================
# API FUNCTION
# =========================

def calculate_priority(sev_input, time_input):
    """
    Calculate fuzzy dispatch priority.

    Args:
        sev_input (float): Reported severity (1–5)
        time_input (float): Estimated travel time in minutes

    Returns:
        float: Priority score (0–100)
    """
    # Clamp inputs to safe ranges
    sev_input = max(0, min(6, sev_input))
    time_input = max(0, min(60, time_input))

    # IMPORTANT: new simulation per call (prevents state leakage)
    sim = ctrl.ControlSystemSimulation(dispatch_ctrl)

    sim.input['severity'] = sev_input
    sim.input['travel_time'] = time_input
    sim.compute()

    return sim.output['priority']


# =========================
# VISUALIZATION (OPTIONAL)
# =========================

def view_membership_functions():
    """Generate membership plots for report figures."""
    severity.view()
    travel_time.view()
    priority.view()
    plt.show()


# =========================
# TESTS
# =========================

if __name__ == "__main__":
    print("Testing Fuzzy System")

    p1 = calculate_priority(5, 5)
    print(f"Severity 5, Time 5 min -> Priority: {p1:.2f}")

    p2 = calculate_priority(1, 55)
    print(f"Severity 1, Time 55 min -> Priority: {p2:.2f}")

    p3 = calculate_priority(3, 25)
    print(f"Severity 3, Time 25 min -> Priority: {p3:.2f}")

    view_membership_functions()
