import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from risk_prediction import get_trained_model, HOTSPOT_PATTERN, generate_historical_data

def generate_figures():
    # Ensure figures directory exists
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # 1. Train Model & Capture History (implicitly done via TensorBoard in get_trained_model)
    # But for the paper figure, we might want to capture loss explicitly or just use the model for heatmaps.
    # Let's train a model and use it.
    print("Training model for figure generation...")
    model = get_trained_model(epochs=200)
    
    # 2. Figure: Risk Heatmap (Spatial) at different times
    # We will plot the predicted risk over the 10x10 map for the 3 distinct time windows.
    
    times_to_plot = [0.15, 0.5, 0.85] # Morning, Day, Evening (centers of the windows)
    time_labels = ["Morning (t=0.15)", "Day (t=0.50)", "Evening (t=0.85)"]
    
    # Create a grid of coordinates
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    xx, yy = np.meshgrid(x, y)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, t in enumerate(times_to_plot):
        # Flatten grid for batch prediction
        grid_points = np.vstack([xx.ravel()/10.0, yy.ravel()/10.0, np.full_like(xx.ravel(), t)]).T
        input_tensor = torch.tensor(grid_points, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = model(input_tensor).numpy().reshape(xx.shape)
            
        # Plot
        ax = axes[i]
        c = ax.contourf(xx, yy, predictions, levels=20, cmap='Reds', vmin=0, vmax=1)
        ax.set_title(f"Predicted Risk: {time_labels[i]}")
        ax.set_xlabel("Map X")
        ax.set_ylabel("Map Y")
        
        # Overlay the 'True' hotspot location for that time
        # Check pattern
        for (t_start, t_end, tid, tx, ty) in HOTSPOT_PATTERN:
            if t_start <= t < t_end:
                ax.scatter(tx, ty, c='blue', marker='x', s=100, label='True Hotspot', linewidths=3)
                ax.legend()
                
    # Adjust layout to make room for colorbar
    plt.tight_layout()
    
    # Add colorbar with proper spacing (fraction controls width, pad controls distance)
    cbar = fig.colorbar(c, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    cbar.set_label("Risk Probability")
    
    save_path = "figures/ann_risk_heatmaps.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # bbox_inches ensures nothing is cut off
    print(f"Saved Risk Heatmap to {save_path}")

    # 3. Figure: Loss Curve (Simulated for visualization)
    # Since we didn't return loss history from get_trained_model, we will just parse the log or 
    # run a quick training loop here specifically to plot the curve.
    print("Generating Loss Curve...")
    inputs, targets = generate_historical_data(n_samples=2000)
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # Quick re-train for 50 epochs to capture the curve
    model_for_plot = type(model)() # Fresh instance
    for epoch in range(50):
        out = model_for_plot(inputs)
        loss = criterion(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='MSE Loss', linewidth=2)
    plt.title("Neural Network Training Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    save_path_loss = "figures/ann_loss_curve.png"
    plt.savefig(save_path_loss, dpi=300)
    print(f"Saved Loss Curve to {save_path_loss}")

if __name__ == "__main__":
    generate_figures()
