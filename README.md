# Ambulance Dispatch Optimization

This project aims to simulate and optimize ambulance dispatching in a city. It provides a foundational environment for developing and testing pathfinding algorithms, such as Genetic Algorithms, to find the most efficient routes for emergency response.

## Current State

The project is currently in the setup phase. The core components that have been implemented are:
1.  **Map Generation**: A script that defines the city map as a graph and can save it to CSV files.
2.  **Dynamic Map**: The map supports real-time changes to road weights to simulate events like traffic jams.
3.  **Simulation Environment**: A framework for managing `Ambulance` agents and spawning random `Emergency` events on the map.

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

## Simulation Environment

The `simulation.py` file sets up the dynamic components of the project:

-   **`Ambulance` Class**: Represents an ambulance agent with properties like its current location, status (`available`, `responding`, `transporting`, `returning`), and a potential patient.
-   **`Emergency` Class**: Represents an emergency event with a specific location and a priority level (from 1 to 5).
-   **`DispatchSimulator` Class**: Manages the entire ecosystem. It is responsible for creating the ambulance fleet and randomly spawning new emergencies on the map at valid locations.

## How to Run

1.  **To view the map and generate the CSV files**:
    ```bash
    python3 ambulance_map.py
    ```
    This script will also demonstrate a random traffic jam by modifying a road's weight.

2.  **To run the dispatch simulation**:
    ```bash
    python3 simulation.py
    ```
    This script will initialize the ambulance fleet and demonstrate several new emergencies spawning in real-time.

## Next Steps

The next major step is to implement the pathfinding algorithm (e.g., a Genetic Algorithm) that will use this environment to calculate the optimal routes for the ambulances to respond to emergencies.
