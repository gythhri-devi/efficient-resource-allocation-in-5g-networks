# 📡 Efficient Resource Allocation in 5G Networks — AMC Module

A neural network-based approach to **Adaptive Modulation and Coding (AMC)** for dynamic bandwidth and power allocation in 5G cellular networks. The model jointly optimises **Quality of Service (QoS)** and **energy efficiency** across multiple cells and users, outperforming fixed and random allocation baselines.

---

## 🗂 Repository Structure

```
amc/
├── resource_allocation.py   # Main model: data generation, training, evaluation
├── app.py                   # Streamlit dashboard for interactive simulation
└── README.md
```

---

## 🔍 Problem Statement

In a multi-cell 5G network, each base station must allocate limited **bandwidth** (MHz) and **transmit power** (W) to dozens of users simultaneously. Traditional approaches either:

- **Over-provision** (waste energy), or
- **Under-provision** (fail to meet user data demands)

This project trains a feedforward neural network to learn per-user allocation policies that satisfy demand while minimising energy consumption.

---

## 🧠 Approach

### Dataset
Synthetic multi-cell network data is generated with:

| Feature | Range |
|---|---|
| User distance from tower | 20 – 700 m |
| SINR (signal quality) | 0.002 – 0.4 |
| CQI (channel quality index) | 1 – 15 |
| User data demand | 5 – 50 Mbps |

Simulation covers **3 cells**, **20 users per cell**, over **50 time steps** (3,000 total samples).

### Model Architecture

A 5-layer feedforward network with a constrained output layer:

```
Input (5 features)
    → Dense(64, ReLU)
    → Dense(128, ReLU)
    → Dense(64, ReLU)
    → Dense(32, ReLU)
    → Dense(2)
    → Lambda (output constraints)
```

Outputs are sigmoid-scaled to physically valid ranges:
- **Bandwidth**: 5 – 20 MHz
- **Power**: 0.4 – 1.0 W

### Custom Loss Function

The model is trained with a dual-objective loss:

```
Loss = QoS_penalty + λ · Energy_penalty
```

Where:
- `QoS_penalty` = mean squared shortfall when throughput < demand
- `Energy_penalty` = mean transmit power
- `λ = 0.05` (tunable energy weight)

Throughput is computed via the Shannon capacity formula:

```
Throughput = Bandwidth × log₂(1 + SINR × Power)
```

---

## 📊 Evaluation

The model is benchmarked against two baselines:

| Method | Description |
|---|---|
| **Neural Model** | Learned adaptive allocation |
| **Fixed Allocation** | Bandwidth = 12.5 MHz, Power = 0.7 W for all users |
| **Random Allocation** | Uniformly random bandwidth and power per user |

Metrics reported:
- **QoS Satisfaction Rate** — fraction of users whose throughput meets demand
- **Average Energy** — mean bandwidth × power product

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib streamlit
```

### Run the model script

```bash
python resource_allocation.py
```

This will train the model and print a comparison table, then display four matplotlib plots:
1. Learned bandwidth and power allocation over the test set
2. Energy consumption per sample
3. Histogram of QoS gap (throughput − demand)
4. Bar chart comparing Model vs Fixed vs Random on QoS and Energy

### Launch the interactive dashboard

```bash
streamlit run app.py
```

The Streamlit app provides:
- Sidebar controls for network size, training hyperparameters, and baseline settings
- Live training progress with real-time loss curve
- KPI cards showing QoS and energy delta vs fixed baseline
- Interactive charts and a summary comparison table

---

## 📈 Results

The neural model adapts allocation per user, achieving higher QoS satisfaction while using less energy than the fixed mid-range policy — demonstrating the value of learned, context-aware resource management.

> Exact numbers vary per run due to random data generation. Set `np.random.seed(42)` is used for reproducibility.

---

## ⚙️ Configuration

Key parameters are easy to adjust at the top of `resource_allocation.py`:

```python
num_users  = 20     # Users per cell
num_cells  = 3      # Number of base stations
time_steps = 50     # Simulation duration

# In energy_qos_loss():
energy_w   = 0.05   # Trade-off weight: higher → more energy-efficient
```

---

## 📚 Concepts

| Term | Meaning |
|---|---|
| **AMC** | Adaptive Modulation and Coding — dynamically adjusting transmission parameters to match channel conditions |
| **SINR** | Signal-to-Interference-plus-Noise Ratio — measures signal quality |
| **CQI** | Channel Quality Indicator — UE feedback on downlink channel conditions |
| **QoS** | Quality of Service — whether a user's data rate requirement is met |
| **Shannon Capacity** | Theoretical maximum data rate: `B · log₂(1 + SNR)` |

---

## 🛠 Tech Stack

- **Python 3.8+**
- **TensorFlow / Keras** — neural network training
- **scikit-learn** — data preprocessing
- **NumPy / Pandas** — data generation and manipulation
- **Matplotlib** — result visualisation
- **Streamlit** — interactive frontend dashboard
