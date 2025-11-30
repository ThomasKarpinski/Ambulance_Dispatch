import csv
import random

"""
This module defines the map for the ambulance dispatch optimization problem.

The map is represented as a graph with a set of locations and an adjacency matrix.
"""

# Define the locations on the map
# Each location is a dictionary with an ID, type, and name.
# Types: 'A' for Ambulance Base, 'H' for Hospital, 'E' for Emergency, 'I' for Intersection
locations = [
    {"id": 0, "type": "A", "name": "Ambulance Base 1"},
    {"id": 1, "type": "H", "name": "City General Hospital"},
    {"id": 2, "type": "H", "name": "Suburban Medical Center"},
    {"id": 3, "type": "E", "name": "Emergency at Downtown"},
    {"id": 4, "type": "I", "name": "Intersection A"},
    {"id": 5, "type": "I", "name": "Intersection B"},
    {"id": 6, "type": "E", "name": "Emergency on Highway"},
]

# Adjacency matrix representing the graph.
# The value at adj_matrix[i][j] is the weight (e.g., travel time) of the road
# between location i and location j.
# A value of 0 indicates no direct road.
# The graph is undirected, so the matrix is symmetric.
adjacency_matrix = [
    # 0(A)  1(H)  2(H)  3(E)  4(I)  5(I)  6(E)
    [0,     0,     0,     0,     2,     0,     0],    # 0: Ambulance Base 1
    [0,     0,     0,     0,     0,     1,     0],    # 1: City General Hospital
    [0,     0,     0,     0,     3,     4,     0],    # 2: Suburban Medical Center
    [0,     0,     0,     0,     1,     0,     5],    # 3: Emergency at Downtown
    [2,     0,     3,     1,     0,     5,     0],    # 4: Intersection A
    [0,     1,     4,     0,     5,     0,     0],    # 5: Intersection B
    [0,     0,     0,     5,     0,     0,     0],    # 6: Emergency on Highway
]

def save_map_to_csv():
    """
    Saves the locations and adjacency matrix to CSV files.
    """
    print("\nSaving map data to CSV files...")

    # Save locations
    with open('locations.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['id', 'type', 'name'])
        # Write data
        for loc in locations:
            writer.writerow([loc['id'], loc['type'], loc['name']])
    print("  - locations.csv created successfully.")

    # Save adjacency matrix
    with open('adjacency_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(adjacency_matrix)
    print("  - adjacency_matrix.csv created successfully.")

def get_location_by_id(location_id):
    """
    Finds a location from the list by its ID.
    """
    for location in locations:
        if location["id"] == location_id:
            return location
    return None

def print_matrix():
    """Prints the adjacency matrix in a readable format."""
    header = "     " + "  ".join([f"{loc['id']:<4}" for loc in locations])
    print(header)
    print("    " + "-" * (len(locations) * 5))

    for i, row in enumerate(adjacency_matrix):
        row_str = f"{i:<4}|"
        for weight in row:
            row_str += f"  {weight:<4}"
        print(row_str)

def update_road_weight(loc1_id, loc2_id, new_weight):
    """
    Updates the weight of the road between two locations in the adjacency matrix.
    Assumes an undirected graph, so it updates the connection in both directions.
    """
    if 0 <= loc1_id < len(adjacency_matrix) and 0 <= loc2_id < len(adjacency_matrix):
        if adjacency_matrix[loc1_id][loc2_id] > 0:
            print(f"\n-> Updating road between '{get_location_by_id(loc1_id)['name']}' and '{get_location_by_id(loc2_id)['name']}'.")
            print(f"   Old weight: {adjacency_matrix[loc1_id][loc2_id]}. New weight: {new_weight}.")
            adjacency_matrix[loc1_id][loc2_id] = new_weight
            adjacency_matrix[loc2_id][loc1_id] = new_weight
            return True
    return False

def simulate_traffic_jam():
    """
    Simulates a random traffic event by finding a random existing road
    and increasing its weight.
    """
    print("\n--- Simulating a random traffic jam... ---")
    while True:
        # Find a random road that actually exists (weight > 0)
        loc1_id = random.randrange(len(adjacency_matrix))
        loc2_id = random.randrange(len(adjacency_matrix))

        if adjacency_matrix[loc1_id][loc2_id] > 0:
            # Increase weight by a random amount from 1 to 3, up to a max of 5
            current_weight = adjacency_matrix[loc1_id][loc2_id]
            if current_weight >= 5: # Don't increase weight if it's already maxed out
                continue
            new_weight = min(5, current_weight + random.randint(1, 3))
            update_road_weight(loc1_id, loc2_id, new_weight)
            break # Exit loop once a road is updated

if __name__ == "__main__":
    # Example of how to use the map data
    print("Ambulance Dispatch Map")
    print("=" * 25)
    
    print("\nLocations:")
    for loc in locations:
        print(f"ID: {loc['id']}, Type: {loc['type']}, Name: {loc['name']}")
        
    print("\nAdjacency Matrix (Original):")
    print_matrix()
    
    # Demonstrate a real-time change
    simulate_traffic_jam()

    print("\nAdjacency Matrix (After Traffic Jam):")
    print_matrix()
    
    # Save the updated map to CSV files
    save_map_to_csv()
