# test.py
from simulation import DispatchSimulator, Ambulance, Emergency
from ambulance_map import find_shortest_path
from ga_dispatcher import GeneticDispatcher
import fuzzy_system

print("=== TEST 1: Fuzzy System ===")
sev = 5
time = 5
priority_score = fuzzy_system.calculate_priority(sev, time)
print(f"Severity {sev}, Travel Time {time} -> Priority Score: {priority_score:.2f}\n")

print("=== TEST 2: Map & Shortest Path ===")
start, end = 0, 3
path, total_time = find_shortest_path(start, end)
print(f"Shortest Path from {start} to {end}: {path}, Total Travel Time: {total_time}\n")

print("=== TEST 3: GA Dispatcher ===")
# Setup some test ambulances
ambulances = [
    Ambulance(id=0, home_base_id=0),
    Ambulance(id=1, home_base_id=0),
]

# Setup some test emergencies
ems = [
    Emergency(id=0, location_id=3, priority=5),
    Emergency(id=1, location_id=6, priority=3),
]

# Run the GA dispatcher
ga = GeneticDispatcher(available_ambulances=ambulances, unassigned_emergencies=ems)
best_assignment = ga.solve()

# Print assignment safely
print("Best GA Assignment:")
for i, amb in enumerate(best_assignment):
    emergency_id = ems[i].id

    if amb is None:
        print(f"Emergency {emergency_id} -> Ambulance None")
    elif isinstance(amb, Ambulance):
        print(f"Emergency {emergency_id} -> Ambulance {amb.id}")
    else:
        # fallback if GA returned an integer or something else
        print(f"Emergency {emergency_id} -> Ambulance {amb}")

