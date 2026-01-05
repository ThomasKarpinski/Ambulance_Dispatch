import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# =========================
# 1. GENERUJEMY DANE (sztuczne)
# =========================

X = []
y = []

for _ in range(5000):
    priority = random.randint(1, 5) / 5
    travel_time = random.uniform(1, 20) / 20
    congestion = random.uniform(1, 5) / 5
    free_ambulances = random.randint(0, 5) / 5
    time_of_day = random.randint(0, 23) / 23

    risk = (
        0.4 * priority +
        0.3 * (1 - travel_time) +
        0.2 * (1 - free_ambulances) +
        0.1 * congestion
    )

    X.append([
        priority,
        travel_time,
        congestion,
        free_ambulances,
        time_of_day
    ])

    y.append(min(max(risk, 0), 1))

X = np.array(X)
y = np.array(y)

# =========================
# 2. BUDUJEMY SIEĆ
# =========================

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(5,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'
)

# =========================
# 3. UCZYMY
# =========================

model.fit(X, y, epochs=20, batch_size=32)

# =========================
# 4. ZAPISUJEMY MODEL
# =========================

model.save("risk_model.h5")
print("✅ Model zapisany jako risk_model.h5")
