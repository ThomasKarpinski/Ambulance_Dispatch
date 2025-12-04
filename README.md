# Ambulance Dispatch Optimization

This project aims to simulate and optimize ambulance dispatching in a city. It provides a foundational environment for developing and testing pathfinding algorithms, such as Genetic Algorithms, to find the most efficient routes for emergency response.

## Current State

The project has evolved from its initial setup phase and now includes a more complete and functional simulation and optimization framework. The key implemented components are:

1.  **Map Generation**: A script that defines the city map as a graph and can save it to CSV files.
2.  **Dynamic Map**: The map supports real-time changes to road weights to simulate events like traffic jams.
3.  **Pathfinding**: Dijkstra's algorithm has been implemented in `ambulance_map.py` to find the shortest path between any two locations on the map.
4.  **Full Simulation Environment**: The `simulation.py` script now manages a complete simulation loop, including:
    *   Ambulance dispatching using a greedy algorithm.
    *   Ambulance movement along paths, patient pickup and dropoff.
    *   Tracking of completed and unresponded emergencies.
5.  **Genetic Algorithm Framework**: `genetic_algorithm.py` contains a basic GA structure for optimizing dispatch strategies. It includes:
    *   A main loop for generations and epochs.
    *   Rudimentary mutation and crossbreeding operators (can be enabled with a `fuzzy` flag).
    *   A fitness evaluation function based on the emergency satisfaction rate.

## Map Structure

The city map is represented as a weighted, undirected graph. This information is primarily managed in `ambulance_map.py` and saved to two CSV files:

### 1. `locations.csv`
This file defines the nodes of our graph. Each line represents a specific point on the map.

-   **Columns**: `id`, `type`, `name`
-   **Location Types**:
    -   `A`: Ambulance Base - Where ambulances start and return.
    -   `H`: Hospital - Where patients are transported.
    -   `E`: Emergency Zone - Pre-defined locations where emergencies can occur.
    -   `I`: Intersection - Junctions that connect the other locations. These are the crossroads of the city network.

### 2. `adjacency_matrix.csv`
This file defines the edges of our graph—the roads connecting the locations.

-   It is a symmetrical matrix where the value at `[row][col]` represents the "weight" or "cost" of traveling the road between the location with `id = row` and the location with `id = col`.
-   **Weight (1-5)**: A lower number (e.g., 1) represents a fast/easy road, while a higher number (e.g., 5) represents a slow/costly road (e.g., due to distance or traffic).
-   **Weight (0)**: A value of zero means there is **no direct road** between the two locations. A path must be found through other connected nodes.

## How to Run

1.  **To view the map and generate the CSV files**:
    ```bash
    python3 ambulance_map.py
    ```
    This script will also demonstrate a random traffic jam by modifying a road's weight.

2.  **To run the dispatch simulation with a greedy approach**:
    ```bash
    python3 simulation.py
    ```
    This script will initialize the ambulance fleet and run a live simulation with dynamic emergency spawning and ambulance dispatching.

3.  **To run the Genetic Algorithm**:
    The `genetic_algorithm.py` file is set up to be imported as a module. To run a demonstration of the GA, you can add a main execution block to the end of the file. See the commit history for an example of how to do this.

4.  **To run the automated tests**:
    ```bash
    python3 -m unittest discover tests
    ```

## Next Steps

The foundational framework is now in place. Future development can focus on enhancing the genetic algorithm and the simulation's realism:

-   **Advanced GA Operators**: Implement more sophisticated mutation and crossbreeding strategies that operate intelligently on ambulance paths and assignments.
-   **Refined Fitness Function**: The fitness function in the GA could be expanded to include metrics like average response time, total travel distance, and resource utilization.
-   **Fuzzy Logic for Dispatch**: Implement the "fuzzy" logic for `reassign_emergencies` in the GA to allow for more complex and optimized dispatch decisions beyond the simple greedy approach.
-   **Simulation Realism**: Add more real-world factors to the simulation, such as varying travel times based on time of day, ambulance availability constraints, and more complex emergency scenarios.
