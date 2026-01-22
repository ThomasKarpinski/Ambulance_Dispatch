import matplotlib.pyplot as plt
from fuzzy_system import severity, travel_time, priority
import os

def save_membership_function(variable, filename, title_override=None):
    # Create the figure
    fig = plt.figure(figsize=(8, 5))
    
    # Use the view method, passing the figure to ensure it plots there
    # Note: skfuzzy view() usually takes specific args or uses plt.gcf()
    # We'll try calling .view() and assuming it plots to the current figure if backend is non-interactive
    
    variable.view()
    
    if title_override:
        plt.title(title_override)
        
    save_path = os.path.join("figures", filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def generate():
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Severity
    # Note: severity.view() might open a window if backend is not configured for off-screen
    # We will try to capture it.
    
    try:
        # Plot Severity
        severity.view()
        plt.title('Membership function for severity')
        plt.savefig('figures/membership_functions_severity.png')
        plt.close()
        print("Saved figures/membership_functions_severity.png")

        # Plot Travel Time
        travel_time.view()
        plt.title('Membership function for travel time')
        plt.savefig('figures/membership_functions_travel_time.png')
        plt.close()
        print("Saved figures/membership_functions_travel_time.png")

        # Plot Priority
        priority.view()
        plt.title('Membership function for priority')
        plt.savefig('figures/membership_functions_priority.png')
        plt.close()
        print("Saved figures/membership_functions_priority.png")
        
    except Exception as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    generate()
