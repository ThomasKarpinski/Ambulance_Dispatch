# Ambulance Dispatch Optimization

An AI-driven simulation framework for optimizing ambulance dispatching in a dynamic urban environment. This project integrates Genetic Algorithms, Fuzzy Logic, and Neural Networks to improve emergency response efficiency.

## 🚀 Key Features

*   **AI-Driven Dispatch**: Advanced ambulance assignment using a **Genetic Algorithm (GA)**.
*   **Fuzzy Logic System**: Intelligent emergency prioritization based on reported severity and estimated travel time.
*   **ANN Risk Prediction**: A **PyTorch-based Artificial Neural Network** that predicts high-risk "hotspots" based on time and location.
*   **Dynamic Simulation**: Real-world scenarios including traffic jams and time-dependent emergency spawning patterns.
*   **Comprehensive Analytics**: Automated experiment runs with performance visualization (response times, utilization, etc.).

## 🏗️ Core Components

### 1. Dispatcher (Genetic Algorithm)
Located in `ga_dispatcher.py`, this component uses a GA to solve the assignment problem:
- **Fitness Function**: Evaluates assignments based on travel time and emergency priority.
- **Fuzzy Integration**: Can use the Fuzzy Logic system to calculate dynamic priorities.
- **Monte Carlo Stability**: Uses multiple runs to stabilize fitness evaluation in stochastic environments.

### 2. Priority System (Fuzzy Logic)
Located in `fuzzy_system.py`, it implements a Mamdani fuzzy inference system:
- **Inputs**: `reported_priority` (0-10) and `travel_time`.
- **Output**: `priority_score` (0-100) used by the GA to rank assignments.

### 3. Risk Prediction (Neural Network)
Located in `risk_prediction.py`, it features a 3-layer MLP:
- **Training**: Learns from historical patterns (time/location) to identify hotspots.
- **Synchronization**: The simulation uses the same "Ground Truth" patterns to spawn emergencies, allowing the ANN to proactively identify high-risk zones.

### 4. Simulation Engine
Located in `simulation.py` and `ambulance_map.py`:
- **Graph-based Map**: City represented as a weighted graph (CSV-based).
- **Dynamic Weights**: Simulates traffic jams by increasing road costs in real-time.
- **State Machine**: Tracks ambulances through `available`, `responding`, `transporting`, and `returning` states.

## 📊 How to Run

### 1. Prepare the Environment
Ensure you have the required dependencies:
```bash
pip install torch numpy pandas matplotlib
```

### 2. Train the Risk Model (ANN)
Generate data and train the neural network to recognize emergency patterns:
```bash
python3 risk_prediction.py
```

### 3. Run Full Experiments
Execute the main experiment suite which compares **GA** vs **GA + Fuzzy** across **static** and **dynamic** maps:
```bash
python3 run.py
```
This will:
- Run 30 trials for each configuration.
- Save results to `experiment_results.csv`.
- Generate performance plots in the `figures/` directory.

### 4. Run Individual Simulations
To see a single simulation run with a greedy approach:
```bash
python3 simulation.py
```

## 📈 Results and Figures

The project generates several visualizations in the `figures/` folder:
- **Response Times**: Boxplots comparing GA and GA+Fuzzy.
- **Utilization**: Ambulance busy-time metrics.
- **Total Distance**: Efficiency of the routes taken.
- **ANN Heatmaps**: Visualization of predicted vs. actual risk zones (in `figures/ANN_figures/`).

## 🛠️ Map Structure

- `locations.csv`: Definitions of Nodes (Ambulance Bases, Hospitals, Emergencies, Intersections).
- `adjacency_matrix.csv`: Definitions of Edges (Roads) and their base travel costs.

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.