import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# 1. Dataset Generation
# -----------------------------
num_users = 20
num_cells = 3
time_steps = 50

np.random.seed(42)

data = []
for t in range(time_steps):
    for cell in range(num_cells):
        for user in range(num_users):
            distance = np.random.uniform(20, 700)
            sinr = np.random.uniform(0.002, 0.4)
            cqi = np.random.randint(1, 16)
            demand = np.random.uniform(5, 50)  # QoS demand in Mbps

            # Energy model: power consumption proportional to SINR requirement
            power = np.random.uniform(0.4, 1.0)
            bandwidth = np.random.uniform(5, 20)

            # Throughput function (simplified Shannon-like)
            throughput = bandwidth * np.log2(1 + sinr * power)
            # Latency placeholder
            latency = demand / (throughput + 1e-6)
            # Energy consumption proportional to power * time
            energy = power * 10  

            data.append([t, cell, user, distance, sinr, cqi, demand, bandwidth, power, throughput, latency, energy])

df = pd.DataFrame(data, columns=[
    'time_step', 'cell_id', 'user_id', 'distance', 'sinr', 'cqi',
    'demand', 'bandwidth', 'power', 'throughput', 'latency', 'energy'
])

# -----------------------------
# 2. Feature Selection
# -----------------------------
# Inputs: distance, sinr, cqi, demand
X = df[['distance', 'sinr', 'cqi', 'demand']].values
# Outputs: resource allocations - bandwidth & power
y = df[['bandwidth', 'power']].values

# Scale features and targets
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# -----------------------------
# 3. Model Definition
# -----------------------------
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='sigmoid')  # output: scaled bandwidth & power
])

# Custom loss: encourage meeting QoS (throughput >= demand) and minimize power
def energy_qos_loss(y_true, y_pred):
    # Inverse scale to actual values
    y_pred_actual = y_pred  # scaled [0,1], already in model range
    bandwidth = y_pred_actual[:, 0]
    power = y_pred_actual[:, 1]

    # Estimate throughput
    sinr = X_train[:, 1]  # SINR
    throughput = bandwidth * tf.math.log(1 + sinr * power) / tf.math.log(2.0)

    demand = X_train[:, 3]  # QoS demand
    qos_penalty = tf.reduce_mean(tf.nn.relu(demand - throughput))  # penalize unmet demand
    energy_penalty = tf.reduce_mean(power)  # minimize power
    return qos_penalty + 0.1 * energy_penalty  # weight energy lightly

# Use standard optimizer with MAE on allocations for simplicity
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# -----------------------------
# 4. Training
# -----------------------------
history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32)

# -----------------------------
# 5. Evaluation
# -----------------------------
y_pred_scaled = model.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test)

mae = np.mean(np.abs(y_true - y_pred), axis=0)
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

print("Test MAE (bandwidth, power):", mae)
print("Test RMSE (bandwidth, power):", rmse)

# -----------------------------
# 6. Example Allocation Check
# -----------------------------
for i in range(5):
    print(f"User {i}: Allocated bandwidth={y_pred[i,0]:.2f}, power={y_pred[i,1]:.2f}, demand={df.iloc[i]['demand']:.2f}")