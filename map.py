import random
import csv
import matplotlib.pyplot as plt

# --------------------------
# 1. Parametry miasta
# --------------------------
rows = 70
cols = 100
num_ambulances = 5
num_emergencies = 10

# --------------------------
# 2. Generowanie węzłów
# --------------------------
nodes = {}  # id -> (x, y)
locations = []  # lista do CSV
node_id = 0

for r in range(rows):
    for c in range(cols):
        nodes[node_id] = (c, -r)  # współrzędne do rysowania
        locations.append({"id": node_id, "type": "I", "name": f"Intersection{node_id}"})
        node_id += 1

# --------------------------
# 3. Losowe ambulansy i wypadki
# --------------------------
all_node_ids = list(nodes.keys())
ambulances = random.sample(all_node_ids, num_ambulances)
emergencies = random.sample([n for n in all_node_ids if n not in ambulances], num_emergencies)
hospitals = random.sample([n for n in all_node_ids if n not in ambulances+emergencies], 3)

# Ustaw typy w locations
for loc in locations:
    if loc["id"] in ambulances:
        loc["type"] = "A"
        loc["name"] = f"AmbulanceBase{loc['id']}"
    elif loc["id"] in hospitals:
        loc["type"] = "H"
        loc["name"] = f"Hospital{loc['id']}"
    elif loc["id"] in emergencies:
        loc["type"] = "E"
        loc["name"] = f"EmergencyZone{loc['id']}"

# --------------------------
# 4. Generowanie połączeń (krawędzie)
# --------------------------
adj_matrix = [[0]* (rows*cols) for _ in range(rows*cols)]
edges = []

for r in range(rows):
    for c in range(cols):
        node = r*cols + c
        # połączenie w prawo
        if c < cols-1:
            right = node + 1
            weight = random.randint(1,5)
            adj_matrix[node][right] = weight
            adj_matrix[right][node] = weight
            edges.append((node, right, weight))
        # połączenie w dół
        if r < rows-1:
            down = node + cols
            weight = random.randint(1,5)
            adj_matrix[node][down] = weight
            adj_matrix[down][node] = weight
            edges.append((node, down, weight))

# --------------------------
# 5. Zapis CSV
# --------------------------
with open("locations.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["id","type","name"])
    writer.writeheader()
    writer.writerows(locations)

with open("adjacency_matrix.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(adj_matrix)

print("Zapisano locations.csv i adjacency_matrix.csv")

# --------------------------
# 6. Wizualizacja
# --------------------------
plt.figure(figsize=(16,12))

# rysowanie krawędzi
for (i,j,w) in edges:
    x = [nodes[i][0], nodes[j][0]]
    y = [nodes[i][1], nodes[j][1]]
    plt.plot(x, y, color='gray', linewidth=0.3)

# rysowanie węzłów
for loc in locations:
    x, y = nodes[loc["id"]]
    if loc["type"] == "A":
        plt.scatter(x, y, s=50, color='red', label='Ambulance' if 'Ambulance' not in plt.gca().get_legend_handles_labels()[1] else "")
    elif loc["type"] == "H":
        plt.scatter(x, y, s=50, color='green', label='Hospital' if 'Hospital' not in plt.gca().get_legend_handles_labels()[1] else "")
    elif loc["type"] == "E":
        plt.scatter(x, y, s=50, color='yellow', label='Emergency' if 'Emergency' not in plt.gca().get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(x, y, s=10, color='blue')

plt.title("Miasto 70x100 - Ambulansy, Szpitale, Wypadki")
plt.axis('off')
plt.legend()
plt.show()
