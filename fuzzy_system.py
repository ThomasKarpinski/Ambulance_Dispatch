import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ==========================================
# 1. Define Fuzzy Variables (Antecedents & Consequent)
# ==========================================

# Input: Severity (Scale 0 to 6 to handle inputs 1-5 comfortably)
# We use a slightly wider range than 1-5 to avoid "edge" issues at the boundaries.
severity = ctrl.Antecedent(np.arange(0, 6.1, 0.1), 'severity')

# Input: Travel Time (Scale 0 to 60 minutes)
travel_time = ctrl.Antecedent(np.arange(0, 61, 1), 'travel_time')

# Output: Priority Score (Scale 0 to 100)
# Higher score = Better candidate for dispatch
priority = ctrl.Consequent(np.arange(0, 101, 1), 'priority')

# ==========================================
# 2. Define Membership Functions (The Shapes)
# ==========================================

# --- Severity Levels ---
# Minor: Covers 1-2. Trapezoid starts high, drops off after 2.
severity['minor'] = fuzz.trapmf(severity.universe, [0, 0, 2, 3])
# Moderate: Peaks at 3.
severity['moderate'] = fuzz.trimf(severity.universe, [2, 3, 4])
# Critical: Covers 4-5. Rises from 3, peaks at 5.
severity['critical'] = fuzz.trapmf(severity.universe, [3, 4, 6, 6])

# --- Travel Time Levels ---
# Short: 0-10 mins is ideal. Fades out by 20.
travel_time['short'] = fuzz.trapmf(travel_time.universe, [0, 0, 10, 20])
# Medium: centered around 25-30 mins.
travel_time['medium'] = fuzz.trimf(travel_time.universe, [10, 30, 50])
# Long: Anything over 40 mins is definitely long.
travel_time['long'] = fuzz.trapmf(travel_time.universe, [40, 50, 60, 60])

# --- Priority Levels (Output) ---
priority['low'] = fuzz.trimf(priority.universe, [0, 0, 40])
priority['standard'] = fuzz.trimf(priority.universe, [20, 50, 80])
priority['urgent'] = fuzz.trimf(priority.universe, [60, 80, 100])
priority['life_threatening'] = fuzz.trimf(priority.universe, [80, 100, 100])

# ==========================================
# 3. Define Fuzzy Rules
# ==========================================
# These rules map inputs to outputs based on your "Motivation" slide.

rule1 = ctrl.Rule(severity['critical'] & travel_time['short'], priority['life_threatening'])
rule2 = ctrl.Rule(severity['critical'] & travel_time['medium'], priority['urgent'])
rule3 = ctrl.Rule(severity['critical'] & travel_time['long'], priority['standard'])

rule4 = ctrl.Rule(severity['moderate'] & travel_time['short'], priority['urgent'])
rule5 = ctrl.Rule(severity['moderate'] & travel_time['medium'], priority['standard'])
rule6 = ctrl.Rule(severity['moderate'] & travel_time['long'], priority['low'])

rule7 = ctrl.Rule(severity['minor'] & travel_time['short'], priority['standard'])
rule8 = ctrl.Rule(severity['minor'] & travel_time['medium'], priority['low'])
rule9 = ctrl.Rule(severity['minor'] & travel_time['long'], priority['low'])


# ... inside fuzzy_system.py ...

# Make Critical cases override distance penalties
rule1 = ctrl.Rule(severity['critical'] & travel_time['short'], priority['life_threatening'])
rule2 = ctrl.Rule(severity['critical'] & travel_time['medium'], priority['life_threatening']) # Upgraded from Urgent
rule3 = ctrl.Rule(severity['critical'] & travel_time['long'], priority['urgent'])             # Upgraded from Standard

# Make Minor cases heavily penalized by distance
rule7 = ctrl.Rule(severity['minor'] & travel_time['short'], priority['standard'])
rule8 = ctrl.Rule(severity['minor'] & travel_time['medium'], priority['low'])
rule9 = ctrl.Rule(severity['minor'] & travel_time['long'], priority['low'])



dispatch_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
dispatch_sim = ctrl.ControlSystemSimulation(dispatch_ctrl)

def calculate_priority(sev_input, time_input):
    """
    Calculates the fuzzy priority score for a single assignment.

    Args:
        sev_input (float): Severity of the call (1-5).
        time_input (float): Estimated travel time in minutes.

    Returns:
        float: A priority score from 0 to 100.
    """
    # Clamp inputs to keep them within the universe boundaries
    # (Prevents crashes if simulation produces weird outliers)
    sev_input = max(0, min(6, sev_input))
    time_input = max(0, min(60, time_input))

    dispatch_sim.input['severity'] = sev_input
    dispatch_sim.input['travel_time'] = time_input

    # Crunch the numbers
    dispatch_sim.compute()

    return dispatch_sim.output['priority']

def view_membership_functions():
    """Helper function to generate plots for your report (W7 deliverable)."""
    severity.view()
    travel_time.view()
    priority.view()
    plt.show()

# ==========================================
# Test Block
# ==========================================
if __name__ == "__main__":
    print("Testing Fuzzy System...")

    # Test Case 1: Critical Call close by
    p1 = calculate_priority(5, 5)
    print(f"Severity 5, Time 5 min -> Priority: {p1:.2f} (Expected: High)")

    # Test Case 2: Minor Call far away
    p2 = calculate_priority(1, 55)
    print(f"Severity 1, Time 55 min -> Priority: {p2:.2f} (Expected: Low)")

    # Test Case 3: Moderate Call, medium distance
    p3 = calculate_priority(3, 25)
    print(f"Severity 3, Time 25 min -> Priority: {p3:.2f} (Expected: Standard)")

    print("\nVisualizing plots (close windows to continue)...")
    view_membership_functions()