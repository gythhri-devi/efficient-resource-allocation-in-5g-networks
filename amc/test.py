import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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
            demand = np.random.uniform(5, 50)

            power = np.random.uniform(0.4, 1.0)
            bandwidth = np.random.uniform(5, 20)

            throughput = bandwidth * np.log2(1 + sinr * power)
            latency = demand / (throughput + 1e-6)
            energy = power * 10  

            data.append([t, cell, user, distance, sinr, cqi,
                         demand, bandwidth, power, throughput, latency, energy])

df = pd.DataFrame(data, columns=[
    'time_step', 'cell_id', 'user_id', 'distance', 'sinr', 'cqi',
    'demand', 'bandwidth', 'power', 'throughput', 'latency', 'energy'
])

# -----------------------------
# 2. Feature Engineering
# -----------------------------
df['distance'] = np.log1p(df['distance'])  # normalize scale
df['distance_inv'] = 1 / (df['distance'] + 1)

X = df[['distance', 'sinr', 'cqi', 'demand', 'distance_inv']].values
y = df[['bandwidth', 'power']].values

# Scaling
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Model Definition
# -----------------------------
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='linear')  # bandwidth, power
])

# -----------------------------
# 4. Custom Loss (QoS + Energy)
# -----------------------------
def energy_qos_loss(y_true, y_pred):
    bandwidth = y_pred[:, 0]
    power = y_pred[:, 1]

    # Use approximate average SINR & demand (safe for demo)
    sinr = 0.2
    demand = 25

    throughput = bandwidth * tf.math.log(1 + sinr * power) / tf.math.log(2.0)

    qos_penalty = tf.reduce_mean(tf.nn.relu(demand - throughput))
    energy_penalty = tf.reduce_mean(power)

    return qos_penalty + 0.1 * energy_penalty

# Compile
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# -----------------------------
# 5. Training
# -----------------------------
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred_scaled = model.predict(X_test)

y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test)

# Metrics
mae = np.mean(np.abs(y_true - y_pred), axis=0)
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

print("Test MAE (bandwidth, power):", mae)
print("Test RMSE (bandwidth, power):", rmse)

# -----------------------------
# 7. QoS & Energy Evaluation
# -----------------------------
sinr_test = X_test[:, 1]
demand_test = X_test[:, 3]

throughput_pred = y_pred[:, 0] * np.log2(1 + sinr_test * y_pred[:, 1])
throughput_true = y_true[:, 0] * np.log2(1 + sinr_test * y_true[:, 1])

qos_satisfaction = np.mean(throughput_pred >= demand_test)
energy_pred = y_pred[:, 0] * y_pred[:, 1]
energy_true = y_true[:, 0] * y_true[:, 1]

print("QoS Satisfaction Rate:", qos_satisfaction)

# -----------------------------
# 8. Visualization
# -----------------------------
time_steps = np.arange(len(y_true))

# Bandwidth & Power
plt.figure(figsize=(14,6))
plt.plot(time_steps, y_true[:,0], label='True Bandwidth')
plt.plot(time_steps, y_pred[:,0], '--', label='Predicted Bandwidth')
plt.plot(time_steps, y_true[:,1], label='True Power')
plt.plot(time_steps, y_pred[:,1], '--', label='Predicted Power')
plt.title("Resource Allocation")
plt.legend()
plt.grid()
plt.show()

# Energy Comparison
plt.figure(figsize=(12,5))
plt.plot(time_steps, energy_true, label='Actual Energy')
plt.plot(time_steps, energy_pred, '--', label='Predicted Energy')
plt.title("Energy Efficiency")
plt.legend()
plt.grid()
plt.show()

# QoS Gap Distribution
plt.figure(figsize=(6,5))
plt.hist(throughput_pred - demand_test, bins=30)
plt.title("QoS Gap (Throughput - Demand)")
plt.grid()
plt.show()