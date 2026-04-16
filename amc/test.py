import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="5G Resource Allocator",
    page_icon="📡",
    layout="wide",
)

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
}

.main { background-color: #0d1117; }

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: #58a6ff;
}
.metric-value.green { color: #3fb950; }
.metric-value.yellow { color: #d29922; }

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

.stButton > button {
    background: #238636;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    padding: 10px 24px;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #2ea043;
}

.status-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #8b949e;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.markdown("# 📡 5G Resource Allocator")
st.markdown("<p style='color:#8b949e; font-size:14px; margin-top:-10px;'>Neural network-based bandwidth & power allocation · QoS + Energy optimization</p>", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("<div class='section-header'>Network Parameters</div>", unsafe_allow_html=True)

    num_users   = st.slider("Users per cell",    5,  50, 20)
    num_cells   = st.slider("Number of cells",   1,   6,  3)
    time_steps  = st.slider("Time steps",       10, 100, 50)

    st.markdown("<div class='section-header'>Training Parameters</div>", unsafe_allow_html=True)
    epochs      = st.slider("Max epochs",        10, 100, 50)
    batch_size  = st.select_slider("Batch size", options=[16, 32, 64, 128], value=32)
    patience    = st.slider("Early stop patience", 3, 15, 5)
    energy_w    = st.slider("Energy penalty weight", 0.01, 0.5, 0.05, step=0.01,
                            help="Higher = more energy-efficient but may reduce QoS")

    st.markdown("<div class='section-header'>Fixed Baseline</div>", unsafe_allow_html=True)
    fixed_bw    = st.slider("Fixed bandwidth (MHz)", 5.0, 20.0, 12.5, step=0.5)
    fixed_pw    = st.slider("Fixed power (W)",       0.4,  1.0,  0.7, step=0.05)

    run_btn = st.button("▶  Run Simulation")

# ─────────────────────────────────────────
# Helper — plot style
# ─────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
GRID_C   = "#21262d"
TEXT_C   = "#8b949e"
BLUE     = "#58a6ff"
GREEN    = "#3fb950"
ORANGE   = "#d29922"
RED      = "#f85149"

def style_ax(ax, title=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_C, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.title.set_color(TEXT_C)
    ax.title.set_fontsize(11)
    ax.title.set_fontfamily("monospace")
    if title:
        ax.set_title(title)
    ax.grid(color=GRID_C, linewidth=0.5)

def make_fig(*args, **kwargs):
    fig = plt.figure(*args, facecolor=DARK_BG, **kwargs)
    return fig

# ─────────────────────────────────────────
# Main logic
# ─────────────────────────────────────────
if run_btn:
    # ── 1. Data generation ──────────────────
    with st.spinner("Generating dataset…"):
        np.random.seed(42)
        data = []
        for t in range(time_steps):
            for cell in range(num_cells):
                for user in range(num_users):
                    distance = np.random.uniform(20, 700)
                    sinr     = np.random.uniform(0.002, 0.4)
                    cqi      = np.random.randint(1, 16)
                    demand   = np.random.uniform(5, 50)
                    data.append([t, cell, user, distance, sinr, cqi, demand])

        df = pd.DataFrame(data, columns=[
            'time_step', 'cell_id', 'user_id',
            'distance', 'sinr', 'cqi', 'demand'
        ])

        df['distance']     = np.log1p(df['distance'])
        df['distance_inv'] = 1 / (df['distance'] + 1)

        X = df[['distance', 'sinr', 'cqi', 'demand', 'distance_inv']].values
        y = df[['sinr', 'demand']].values

        X_scaler = MinMaxScaler()
        X_scaled = X_scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

    st.markdown(f"<div class='status-box'>✓ Dataset ready — {len(df):,} samples · {len(X_train):,} train · {len(X_test):,} test</div>",
                unsafe_allow_html=True)

    # ── 2. Model ────────────────────────────
    def output_constraints(x):
        bandwidth = 5  + 15  * tf.sigmoid(x[:, 0])
        power     = 0.4 + 0.6 * tf.sigmoid(x[:, 1])
        return tf.stack([bandwidth, power], axis=1)

    model = Sequential([
        Dense(64,  activation='relu', input_dim=X_train.shape[1]),
        Dense(128, activation='relu'),
        Dense(64,  activation='relu'),
        Dense(32,  activation='relu'),
        Dense(2),
        Lambda(output_constraints)
    ])

    def energy_qos_loss(y_true, y_pred):
        bandwidth  = y_pred[:, 0]
        power      = y_pred[:, 1]
        sinr       = y_true[:, 0]
        demand     = y_true[:, 1]
        throughput = bandwidth * tf.math.log(1 + sinr * power) / tf.math.log(2.0)
        qos_pen    = tf.reduce_mean(tf.square(tf.nn.relu(demand - throughput)))
        energy_pen = tf.reduce_mean(power)
        return qos_pen + energy_w * energy_pen

    model.compile(optimizer='adam', loss=energy_qos_loss)

    # ── 3. Training with live progress ──────
    st.markdown("<div class='section-header'>Training Progress</div>", unsafe_allow_html=True)
    progress_bar   = st.progress(0)
    loss_col, vloss_col = st.columns(2)
    loss_ph   = loss_col.empty()
    vloss_ph  = vloss_col.empty()
    chart_ph  = st.empty()

    train_losses, val_losses = [], []

    class StreamlitCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            tl = logs.get('loss', 0)
            vl = logs.get('val_loss', 0)
            train_losses.append(tl)
            val_losses.append(vl)
            pct = int((epoch + 1) / epochs * 100)
            progress_bar.progress(min(pct, 100))
            loss_ph.metric("Train loss",  f"{tl:.4f}")
            vloss_ph.metric("Val loss",   f"{vl:.4f}")
            # live chart
            if len(train_losses) > 1:
                fig, ax = plt.subplots(figsize=(8, 2.5))
                fig.patch.set_facecolor(DARK_BG)
                style_ax(ax, "Loss curve")
                ax.plot(train_losses, color=BLUE,   lw=1.5, label="train")
                ax.plot(val_losses,   color=ORANGE, lw=1.5, label="val", linestyle="--")
                ax.legend(fontsize=9, labelcolor=TEXT_C, facecolor=PANEL_BG)
                ax.set_xlabel("Epoch", color=TEXT_C, fontsize=9)
                chart_ph.pyplot(fig)
                plt.close(fig)

    early_stop = EarlyStopping(patience=patience, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, StreamlitCallback()],
        verbose=0
    )
    progress_bar.progress(100)
    st.markdown(f"<div class='status-box'>✓ Training complete — {len(train_losses)} epochs</div>",
                unsafe_allow_html=True)

    # ── 4. Evaluation ───────────────────────
    y_pred        = model.predict(X_test, verbose=0)
    bw_pred       = y_pred[:, 0]
    pw_pred       = y_pred[:, 1]
    sinr_test     = y_test[:, 0]
    demand_test   = y_test[:, 1]
    tput_pred     = bw_pred * np.log2(1 + sinr_test * pw_pred)
    qos_model     = float(np.mean(tput_pred >= demand_test))
    energy_model  = float(np.mean(bw_pred * pw_pred))

    # Fixed baseline
    bw_fixed      = np.full_like(sinr_test, fixed_bw)
    pw_fixed      = np.full_like(sinr_test, fixed_pw)
    tput_fixed    = bw_fixed * np.log2(1 + sinr_test * pw_fixed)
    qos_fixed     = float(np.mean(tput_fixed >= demand_test))
    energy_fixed  = float(np.mean(bw_fixed * pw_fixed))

    # Random baseline
    np.random.seed(42)
    bw_rand       = np.random.uniform(5, 20, len(sinr_test))
    pw_rand       = np.random.uniform(0.4, 1.0, len(sinr_test))
    tput_rand     = bw_rand * np.log2(1 + sinr_test * pw_rand)
    qos_rand      = float(np.mean(tput_rand >= demand_test))
    energy_rand   = float(np.mean(bw_rand * pw_rand))

    # ── 5. KPI Cards ────────────────────────
    st.divider()
    st.markdown("<div class='section-header'>Results</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Model QoS</div>
            <div class='metric-value green'>{qos_model:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Model Energy</div>
            <div class='metric-value'>{energy_model:.2f} W·MHz</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        delta_qos = qos_model - qos_fixed
        color = "green" if delta_qos >= 0 else "red"
        sign  = "+" if delta_qos >= 0 else ""
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>vs Fixed QoS</div>
            <div class='metric-value {color}'>{sign}{delta_qos:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        delta_e = energy_model - energy_fixed
        color = "green" if delta_e <= 0 else "yellow"
        sign  = "+" if delta_e >= 0 else ""
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>vs Fixed Energy</div>
            <div class='metric-value {color}'>{sign}{delta_e:.2f}</div>
        </div>""", unsafe_allow_html=True)

    # ── 6. Charts ───────────────────────────
    st.divider()
    col_l, col_r = st.columns(2)

    # Allocation plot
    with col_l:
        st.markdown("<div class='section-header'>Learned Allocation (test set)</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor(DARK_BG)
        style_ax(ax)
        x_idx = np.arange(len(bw_pred))
        ax.plot(x_idx, bw_pred, color=BLUE,   lw=1,   label="Bandwidth (MHz)", alpha=0.85)
        ax.plot(x_idx, pw_pred, color=GREEN,  lw=1,   label="Power (W)",       alpha=0.85)
        ax.axhline(fixed_bw, color=BLUE,  lw=0.8, linestyle=":", alpha=0.4, label="Fixed BW")
        ax.axhline(fixed_pw, color=GREEN, lw=0.8, linestyle=":", alpha=0.4, label="Fixed PW")
        ax.legend(fontsize=8, labelcolor=TEXT_C, facecolor=PANEL_BG, loc="upper right")
        ax.set_xlabel("Sample", color=TEXT_C, fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

    # QoS gap histogram
    with col_r:
        st.markdown("<div class='section-header'>QoS Gap (Throughput − Demand)</div>", unsafe_allow_html=True)
        gap = tput_pred - demand_test
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor(DARK_BG)
        style_ax(ax)
        ax.axvline(0, color=RED, lw=1.2, linestyle="--", alpha=0.7)
        ax.hist(gap[gap < 0], bins=25, color=RED,   alpha=0.7, label="Under-served")
        ax.hist(gap[gap >= 0], bins=25, color=GREEN, alpha=0.7, label="Satisfied")
        ax.legend(fontsize=8, labelcolor=TEXT_C, facecolor=PANEL_BG)
        ax.set_xlabel("Gap (Mbps)", color=TEXT_C, fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

    # Baseline comparison bar chart
    st.markdown("<div class='section-header'>Baseline Comparison</div>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor(DARK_BG)

    labels = ["Model", "Fixed", "Random"]
    qos_vals    = [qos_model, qos_fixed, qos_rand]
    energy_vals = [energy_model, energy_fixed, energy_rand]
    colors      = [BLUE, ORANGE, RED]

    for ax, vals, title, unit in zip(
        axes,
        [qos_vals, energy_vals],
        ["QoS Satisfaction Rate", "Average Energy (W·MHz)"],
        ["%", ""]
    ):
        style_ax(ax, title)
        bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor=DARK_BG, linewidth=0.5)
        for bar, val in zip(bars, vals):
            label = f"{val:.1%}" if unit == "%" else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals) * 0.02,
                    label, ha='center', va='bottom',
                    color=TEXT_C, fontsize=9, fontfamily="monospace")
        ax.tick_params(colors=TEXT_C)
        ax.set_ylim(0, max(vals) * 1.2)

    fig.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close(fig)

    # ── 7. Summary table ────────────────────
    st.divider()
    st.markdown("<div class='section-header'>Summary Table</div>", unsafe_allow_html=True)
    summary = pd.DataFrame({
        "Method":          ["Neural Model", "Fixed Allocation", "Random Allocation"],
        "QoS Satisfaction":  [f"{qos_model:.1%}",    f"{qos_fixed:.1%}",   f"{qos_rand:.1%}"],
        "Avg Energy (W·MHz)":[f"{energy_model:.3f}", f"{energy_fixed:.3f}",f"{energy_rand:.3f}"],
        "Avg Bandwidth (MHz)":[f"{bw_pred.mean():.2f}", f"{fixed_bw:.2f}", f"{bw_rand.mean():.2f}"],
        "Avg Power (W)":     [f"{pw_pred.mean():.3f}", f"{fixed_pw:.3f}",  f"{pw_rand.mean():.3f}"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

else:
    # Landing state
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px; color: #8b949e;'>
        <div style='font-size: 48px; margin-bottom: 16px;'>📡</div>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 18px; color: #58a6ff; margin-bottom: 12px;'>
            Ready to simulate
        </div>
        <div style='font-size: 14px; max-width: 480px; margin: 0 auto; line-height: 1.7;'>
            Configure the network parameters and training settings in the sidebar,
            then click <strong style='color: #3fb950;'>▶ Run Simulation</strong> to train the neural network
            and compare it against fixed and random baselines.
        </div>
    </div>
    """, unsafe_allow_html=True)
