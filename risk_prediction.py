# Instruction: Update data generation to follow a strict time-location pattern.
# This ensures the ANN learns a "Ground Truth" that we can also enforce in the simulation.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import shutil

# Hardcoded pattern for synchronization between Training and Simulation
# (Location ID, x, y)
# ID 3: Downtown (3, 4)
# ID 6: Highway (10, 3)
# ID 4: Intersection A (2, 2)
HOTSPOT_PATTERN = [
    (0.0, 0.33, 3, 3.0, 4.0),   # Time 0.00-0.33 -> ID 3
    (0.33, 0.66, 6, 10.0, 3.0), # Time 0.33-0.66 -> ID 6
    (0.66, 1.01, 4, 2.0, 2.0),  # Time 0.66-1.00 -> ID 4
]

def get_ground_truth_risk(loc_id, norm_time):
    """
    Returns the 'True' risk probability for a location at a specific time.
    Used by:
    1. Data Generator (to train ANN)
    2. Simulation (to spawn emergencies matching the pattern)
    """
    # Default low risk
    risk = 0.1
    
    for (t_start, t_end, target_id, _, _) in HOTSPOT_PATTERN:
        if t_start <= norm_time < t_end:
            if loc_id == target_id:
                return 0.9 # High risk at the active hotspot
            
    return risk

def generate_historical_data(n_samples=10000):
    """
    Generates training data based on the HOTSPOT_PATTERN.
    """
    X_list = []
    y_list = []
    
    # Map dimensions for normalization (assuming 10x10 roughly)
    MAX_X = 10.0
    MAX_Y = 10.0
    
    for _ in range(n_samples):
        # 1. Pick a random time
        t = np.random.rand()
        
        # 2. Pick a location ID based on the pattern logic
        # We simulate the "World" choosing where to put an event
        active_hotspot = None
        for (t_start, t_end, target_id, tx, ty) in HOTSPOT_PATTERN:
            if t_start <= t < t_end:
                active_hotspot = (target_id, tx, ty)
                break
        
        # 80% chance to be at the hotspot, 20% noise (random valid-ish spot)
        if active_hotspot and np.random.rand() < 0.8:
            _, true_x, true_y = active_hotspot
            # Add small jitter to coordinate
            x = true_x + np.random.normal(0, 0.5)
            y = true_y + np.random.normal(0, 0.5)
            risk_label = 1.0 # High risk event occurred here
        else:
            # Random location
            x = np.random.uniform(0, MAX_X)
            y = np.random.uniform(0, MAX_Y)
            risk_label = 0.0 # Just noise / no event
            
        # Normalize inputs
        nx = x / MAX_X
        ny = y / MAX_Y
        
        X_list.append([nx, ny, t])
        y_list.append([risk_label])
        
    return torch.tensor(X_list, dtype=torch.float32), torch.tensor(y_list, dtype=torch.float32)


# --- Neural Network Class (Unchanged) ---
class RiskAssessmentNet(nn.Module):
    def __init__(self):
        super(RiskAssessmentNet, self).__init__()
        # Inputs: location_x, location_y, time_of_day
        self.fc1 = nn.Linear(3, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.output(x))
        return x


def train_risk_model(model, inputs, targets, epochs=100, log_dir="runs/risk_experiment_improved"):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    
    # Log the network graph for TensorBoard
    # We need a dummy input of shape (1, 3) because 'inputs' might be huge
    dummy_input = torch.rand(1, 3)
    writer.add_graph(model, dummy_input)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005) # Slightly lower LR
    
    for epoch in range(epochs):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log loss to TensorBoard
        writer.add_scalar('Training Loss', loss.item(), epoch)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    writer.close()

def get_trained_model(epochs=100):
    print("Training Improved Risk Prediction Model (Aligned with Simulation)...")
    # Generate MORE data for better pattern recognition
    inputs, targets = generate_historical_data(n_samples=5000)
    model = RiskAssessmentNet()
    train_risk_model(model, inputs, targets, epochs=epochs)
    model.eval()
    return model

if __name__ == "__main__":
    # Test the generator
    model = get_trained_model(epochs=50)
    # Test inference: Time 0.1 should favor Downtown (3,4) -> norm(0.3, 0.4)
    t = 0.1
    test_pt = torch.tensor([[0.3, 0.4, t]])
    print(f"Risk at Downtown at t={t}: {model(test_pt).item():.4f}")