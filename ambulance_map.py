import csv
import random
import heapq
import copy
import math

"""
Ambulance Dispatch Map Module

Defines:
- Locations with coordinates
- Road network (adjacency matrix)
- Shortest path search (Dijkstra)
- Stochastic travel times (uncertainty)
- Traffic simulation
- Map reset for fair experiments
"""

# =========================
# LOCATION DEFINITIONS
# =========================
# Types:
# A = Ambulance Base
# H = Hospital
# E = Emergency hotspot
# I = Intersection

locations = [
    {"id": 0, "type": "A", "name": "Ambulance Base 1", "x": 0, "y": 0},
    {"id": 1, "type": "H", "name": "City General Hospital", "x": 6, "y": 1},
    {"id": 2, "type": "H", "name": "Suburban Medical Center", "x": 9, "y": 6},
    {"id": 3, "type": "E", "name": "Emergency at Downtown", "x": 3, "y": 4},
    {"id": 4, "type": "I", "name": "Intersection A", "x": 2, "y": 2},
    {"id": 5, "type": "I", "name": "Intersection B", "x": 6, "y": 4},
    {"id": 6, "type": "E", "name": "Emergency on Highway", "x": 10, "y": 3},
]

# =========================
# ADJACENCY MATRIX
# =========================
# 0 means no road
adjacency_matrix = [
    # 0  1  2  3  4  5  6
    [0, 0, 0, 0, 2, 0, 0],  # 0 Base
    [0, 0, 0, 0, 0, 1, 0],  # 1 Hospital
    [0, 0, 0, 0, 3, 4, 0],  # 2 Hospital
    [0, 0, 0, 0, 1, 0, 5],  # 3 Emergency
    [2, 0, 3, 1, 0, 5, 0],  # 4 Intersection
    [0, 1, 4, 0, 5, 0, 0],  # 5 Intersection
    [0, 0, 0, 5, 0, 0, 0],  # 6 Emergency
]

# Save original matrix for resets
_original_matrix = copy.deepcopy(adjacency_matrix)

# =========================
# UTILITY FUNCTIONS
# =========================

def get_location_by_id(location_id):
    for loc in locations:
        if loc["id"] == location_id:
            return loc
    return None


def euclidean_distance(id1, id2):
    """Straight-line distance (used for fuzzy logic if needed)."""
    l1 = get_location_by_id(id1)
    l2 = get_location_by_id(id2)
    return math.sqrt((l1["x"] - l2["x"]) ** 2 + (l1["y"] - l2["y"]) ** 2)


# =========================
# TRAVEL TIME (DETERMINISTIC + STOCHASTIC)
# =========================

def get_travel_time(u, v, stochastic=False):
    base_time = adjacency_matrix[u][v]
    if base_time == 0:
        return float("inf")

    if stochastic:
        noise = random.uniform(0.8, 1.3)  # ±30% delay
        return base_time * noise

    return base_time


# =========================
# SHORTEST PATH (DIJKSTRA)
# =========================

def find_shortest_path(start_id, end_id, stochastic=False):
    num_locations = len(adjacency_matrix)
    distances = {i: float("inf") for i in range(num_locations)}
    previous = {i: None for i in range(num_locations)}

    distances[start_id] = 0
    pq = [(0, start_id)]

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        if current_dist > distances[current_node]:
            continue

        if current_node == end_id:
            break

        for neighbor in range(num_locations):
            if adjacency_matrix[current_node][neighbor] > 0:
                weight = get_travel_time(current_node, neighbor, stochastic)
                new_dist = current_dist + weight

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))

    # Reconstruct path
    path = []
    node = end_id
    while node is not None:
        path.insert(0, node)
        node = previous[node]

    if path and path[0] == start_id:
        return path, distances[end_id]

    return None, float("inf")


# =========================
# TRAFFIC & RESET
# =========================

def simulate_traffic_jam():
    """Randomly increases travel time on one existing road."""
    while True:
        i = random.randrange(len(adjacency_matrix))
        j = random.randrange(len(adjacency_matrix))

        if adjacency_matrix[i][j] > 0:
            if adjacency_matrix[i][j] < 5:
                adjacency_matrix[i][j] += random.randint(1, 2)
                adjacency_matrix[j][i] = adjacency_matrix[i][j]
            break


def reset_map():
    """Resets the map to its original state (important for experiments)."""
    global adjacency_matrix
    adjacency_matrix = copy.deepcopy(_original_matrix)


# =========================
# CSV EXPORT (OPTIONAL)
# =========================

def save_map_to_csv():
    with open("locations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "type", "name", "x", "y"])
        for loc in locations:
            writer.writerow([loc["id"], loc["type"], loc["name"], loc["x"], loc["y"]])

    with open("adjacency_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(adjacency_matrix)


# =========================
# DEMO
# =========================

if __name__ == "__main__":
    print("Shortest path (deterministic):")
    print(find_shortest_path(0, 3))

    print("\nShortest path (stochastic):")
    print(find_shortest_path(0, 3, stochastic=True))

    print("\nSimulating traffic...")
    simulate_traffic_jam()
    print(find_shortest_path(0, 3, stochastic=True))

    print("\nResetting map...")
    reset_map()
    print(find_shortest_path(0, 3))

