## Overview

This project implements an AI-driven AIOps solution for telecom network monitoring using Machine Learning and Deep Learning techniques. It analyzes historical Key Performance Indicator (KPI) data to detect abnormal network behavior, forecast KPI trends, and support proactive network maintenance.

The project uses:

- Isolation Forest for unsupervised anomaly detection
- Long Short-Term Memory (LSTM) Neural Network for time-series anomaly detection
- Python, Pandas, Scikit-learn, PyTorch, and Matplotlib
# Objectives

- Analyze telecom KPI time-series data.
- Detect abnormal network conditions.
- Forecast future KPI trends.
- Improve Quality of Service (QoS).
- Reduce manual network monitoring efforts.

# Dataset

The dataset contains 30 days of telecom KPI data collected every 5 minutes.

### Features

| Column | Description |
|---------|-------------|
| Timestamp | Date and time |
| Cell_ID | Cell Tower ID |
| RSRP | Reference Signal Received Power |
| SINR | Signal to Interference plus Noise Ratio |
| Latency | Network latency (ms) |
| Throughput | Data throughput (Mbps) |
| Packet_Loss | Packet loss (%) |
| Connected_Users | Number of active users |
| QoS_Label | Service Quality |
| KPI_Label | Network Status |

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- PyTorch
- Joblib
- OpenPyXL

# Project Structure

```
telecom-network-aiops/

│
|── telecom_kpi.csv
│
├── models/
│   └── lstm_model.pth
│
├── task1_data_exploration.py
├── task2_visualization.py
├── task3_isolation_forest.py
├── task4_lstm.py
│
├── lstm_detected_anomalies.csv
├── detected_anomalies_only.csv
│
├── README.md

# Task 1 – Time Series Data Exploration
## Objectives

- Load KPI dataset
- Convert Timestamp into datetime format
- Sort data chronologically
- Detect missing timestamps
- Identify missing KPI values
- Analyze daily KPI trends
- Analyze weekly KPI trends

### Output

- Daily KPI graphs
- Weekly KPI graphs
- Statistical summary

# Task 2 – KPI Visualization Dashboard

## Objectives

Visualize telecom network KPIs.

### Graphs

- Throughput Trend
- Latency Trend
- SINR Trend
- RSRP Trend
- Packet Loss Trend
- Connected Users Trend
- Daily KPI Trends
- Weekly KPI Trends
- Hourly Traffic Pattern

### Output

Multiple KPI visualization plots for network analysis.

# Task 3 – Isolation Forest Anomaly Detection

## Objective

Detect abnormal KPI behavior using an unsupervised machine learning algorithm.

## Steps

- Load KPI dataset
- Select network KPIs
- Normalize features
- Train Isolation Forest
- Detect anomalies
- Export anomaly results

### Output

- isolation_forest_model.pkl
- scaler.pkl
- detected_anomalies.csv

# Task 4 – LSTM-Based Time Series Anomaly Detection

## Objective

Train an LSTM model to learn normal telecom KPI behavior and detect anomalies based on prediction error.

## Workflow

1. Load KPI data
2. Normalize features
3. Create sliding windows
4. Build LSTM model
5. Train model
6. Predict next KPI values
7. Compute prediction error
8. Detect anomalies
9. Visualize anomalous KPIs
10. Save anomaly report

### LSTM Architecture

- Input Layer
- 2 LSTM Layers
- Dropout Layer (0.2)
- Fully Connected Layer

### Hyperparameters

| Parameter | Value |
|------------|-------|
| Hidden Units | 64 |
| Layers | 2 |
| Sequence Length | 20 |
| Epochs | 20 |
| Batch Size | 64 |
| Optimizer | Adam |
| Loss Function | Mean Squared Error |

### Output Files

- models/lstm_model.pth
- lstm_detected_anomalies.csv
- detected_anomalies_only.csv

# Anomaly Detection Process

The LSTM predicts the next KPI values based on historical observations.

Prediction Error is calculated using Mean Squared Error (MSE).

Samples with prediction error above the 95th percentile threshold are classified as anomalies.

# Visualizations

The project generates:

- Training Loss Curve
- Throughput Anomaly Plot
- Latency Anomaly Plot
- SINR Anomaly Plot
- RSRP Anomaly Plot
- Packet Loss Anomaly Plot
- Connected Users Anomaly Plot

# Results

The project successfully:

- Explores telecom KPI time-series data
- Visualizes network behavior
- Detects anomalies using Isolation Forest
- Detects anomalies using LSTM
- Generates anomaly reports
- Produces KPI visualizations

# Future Improvements

- LSTM Autoencoder
- Transformer-based anomaly detection
- Real-time streaming using Kafka
- Streamlit dashboard
- Root Cause Analysis
- KPI Forecasting
- Network Quality Prediction

# Author
Vibhuti Chaddha
