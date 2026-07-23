# AIOps – Network Anomaly Detection & Root Cause Analysis

## Overview

This project implements an **AI-driven telecom network monitoring and AIOps system** using Machine Learning, Deep Learning, Time-Series Analysis, and Streamlit.

The system analyzes telecom Key Performance Indicators (KPIs) to:

- Monitor network performance
- Detect abnormal network behavior
- Detect time-series anomalies
- Tune anomaly detection thresholds
- Forecast future network traffic
- Perform Root Cause Analysis (RCA)
- Predict Quality of Service / Quality of Experience (QoS/QoE)
- Label network conditions as Normal, Warning, or Critical
- Visualize network health through an interactive Streamlit dashboard

# Project Objectives

The primary objective is to build an intelligent telecom monitoring system capable of identifying network degradation before it significantly impacts users.

The project combines:

- Time-Series Analysis
- Isolation Forest
- LSTM Neural Networks
- Random Forest Classification
- Rule-Based Root Cause Analysis
- Traffic Forecasting
- Streamlit Dashboard

# Dataset

The project uses a telecom KPI time-series dataset containing approximately 30 days of network measurements.

The data is collected at regular time intervals for multiple network cells.

## Main Features

| Feature | Description |
|---|---|
| Timestamp | Date and time of KPI measurement |
| Cell_ID | Unique network cell identifier |
| RSRP | Reference Signal Received Power |
| SINR | Signal-to-Interference-plus-Noise Ratio |
| Latency | Network delay in milliseconds |
| Throughput | Network throughput in Mbps |
| Packet_Loss | Percentage of lost packets |
| Connected_Users | Number of users connected to the cell |
| QoS_Label | Good / Fair / Poor network quality |
| KPI_Label | Normal / Warning / Critical network condition |

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Matplotlib
- Plotly
- Streamlit
- Joblib

# Task 1: Time-Series Data Exploration

The first task analyzes telecom KPI data as a time series.

## Activities

- Load the KPI dataset
- Convert Timestamp to datetime
- Sort records chronologically
- Identify missing timestamps
- Detect missing KPI values
- Analyze daily KPI trends
- Analyze weekly KPI trends

## Outcome

Time-series analysis provides an understanding of network behavior over different time periods and helps identify traffic peaks and performance degradation.

# Task 2: KPI Visualization Dashboard

Visualizations were developed for the major network KPIs.

## KPI Trends

- Throughput over time
- Latency over time
- SINR trend
- RSRP trend
- Packet Loss trend
- Connected Users trend

Additional analysis includes:

- Daily average KPIs
- Hourly traffic patterns
- Weekly traffic comparison

These visualizations make it easier to understand network behavior and identify abnormal periods.

# Task 3: Isolation Forest Anomaly Detection

Isolation Forest was implemented for unsupervised anomaly detection.

## Workflow

```text
KPI Dataset
     ↓
Feature Selection
     ↓
Normalization
     ↓
Isolation Forest
     ↓
Anomaly Score
     ↓
Normal / Anomaly
```
## Features

The model analyzes:

- RSRP
- SINR
- Latency
- Throughput
- Packet Loss
- Connected Users

Isolation Forest is suitable because it can identify abnormal observations without requiring manually labeled anomaly data.

# Task 4: LSTM-Based Anomaly Detection

An LSTM neural network was developed to detect anomalies based on temporal KPI patterns.

## Workflow

```text
Time-Series KPIs
       ↓
Normalization
       ↓
Sliding Windows
       ↓
LSTM Network
       ↓
KPI Prediction
       ↓
Prediction Error
       ↓
Anomaly Detection
```

## Model Configuration

| Parameter | Value |
|---|---:|
| Sequence Length | 20 |
| Hidden Units | 64 |
| LSTM Layers | 2 |
| Batch Size | 64 |
| Epochs | 20 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | MSE |

The model learns temporal network behavior and uses prediction error to identify unusual sequences.

# Task 5: Threshold Tuning

Different anomaly thresholds were evaluated to determine an appropriate balance between anomaly detection and false alarms.

## Thresholds

| Threshold | Behaviour |
|---|---|
| Low – 90th percentile | High sensitivity |
| Medium – 95th percentile | Balanced |
| High – 99th percentile | Detects mainly severe anomalies |

## Comparison

The thresholds are evaluated using:

- Number of detected anomalies
- False positives
- Missed anomalies

The medium threshold provides a practical balance for the implemented system.

# 🔮 Task 6: Network Traffic Forecasting

An LSTM forecasting model was developed to predict future network throughput.

## Forecast Horizons

- Next 24 hours
- Next 7 days

## Accuracy Metrics

The forecasting model can be evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

Traffic forecasting can help network operators anticipate high-load periods and allocate network resources proactively.

# Task 7: Root Cause Analysis

The Root Cause Analysis module analyzes detected anomalies and determines probable causes using KPI-based rules.

| Observation | Probable Root Cause | Recommended Action |
|---|---|---|
| Low RSRP | Poor radio coverage | Optimize antenna tilt |
| Low SINR | High interference | Review neighboring cells |
| High Latency | Core network congestion | Check UPF utilization |
| High Packet Loss | Backhaul issue | Inspect transport network |
| Low Throughput | Heavy traffic load | Enable load balancing |
| High Connected Users | Cell overload | Rebalance users / increase capacity |

The RCA module generates:

```text
root_cause_analysis.csv
root_cause_summary.csv
```

# Task 8: QoS/QoE Prediction

A Random Forest classifier predicts network quality using:

- RSRP
- SINR
- Latency
- Packet Loss
- Throughput

## Classes

```text
Good
Fair
Poor
```

## Evaluation

The classifier is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

The trained model is stored as:

```text
qos_prediction_model.pkl
```
# Task 9: Data Labeling Strategy

A rule-based labeling strategy classifies telemetry records as:

- Normal
- Warning
- Critical

## KPI Thresholds

| KPI | Normal | Warning | Critical |
|---|---|---|---|
| RSRP | > -90 dBm | -90 to -105 | < -105 |
| SINR | > 20 dB | 10–20 | < 10 |
| Latency | < 20 ms | 20–50 | > 50 |
| Packet Loss | < 1% | 1–3% | > 3% |

The labeled dataset is exported as:

```text
telecom_kpi_labeled.csv
```

# Task 10: Final Streamlit Dashboard

A Streamlit-based dashboard integrates the outputs of the complete AIOps pipeline.

## Dashboard Features

### Live KPI Trends

Interactive visualization of:

- RSRP
- SINR
- Latency
- Throughput
- Packet Loss
- Connected Users

### Detected Anomalies

Displays anomalies identified by the anomaly detection models.

### Forecast Graphs

Displays network throughput forecasts for:

- Next 24 hours
- Next 7 days

### QoS/QoE Prediction

Displays predicted network quality:

```text
Good / Fair / Poor
```

### Root Cause Analysis

Displays:

```text
Observation
↓
Possible Root Cause
↓
Recommended Action
```

### Cell-Wise KPI Status

Shows network status for individual cells.

### KPI Threshold Alerts

Automatically generates warnings when KPI values exceed critical thresholds.

# Task 11: Technical Report

A technical report summarizes the complete project.

The report contains:

1. Problem Statement
2. Dataset Description
3. Time-Series Analysis
4. Isolation Forest Results
5. LSTM Model Summary
6. Traffic Forecasting Results
7. Root Cause Analysis
8. QoS Prediction Results
9. Conclusions
10. Future Improvements

# Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

Move into the project:

```bash
cd AIOps
```

## 2. Create Virtual Environment

```bash
python3 -m venv venv
```

Activate it on Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install pandas numpy matplotlib scikit-learn torch streamlit plotly joblib
```
# Running Individual Tasks

Run the scripts sequentially:

```bash
python task1_time_series.py
python task2_visualization.py
python task3_isolation_forest.py
python task4_lstm.py
python task5_threshold_tuning.py
python task6_network_forecasting.py
python task7_root_cause_analysis.py
python task8_qos_prediction.py
python task9_data_labeling.py
```
# Running the Dashboard

Activate the virtual environment:

```bash
source venv/bin/activate
```

Start Streamlit:

```bash
python -m streamlit run final_dashboard.py
```

Then open the address displayed by Streamlit in your browser.

# Complete AIOps Pipeline

```text
               Telecom KPI Dataset
                        │
                        ▼
              Time-Series Analysis
                        │
                        ▼
                KPI Visualization
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
       Isolation Forest          LSTM
              │                   │
              └─────────┬─────────┘
                        ▼
                Anomaly Detection
                        │
                        ▼
                Threshold Tuning
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
      Traffic Forecasting    Root Cause Analysis
                                     │
                                     ▼
                              Recommended Action

               KPI Dataset
                    │
                    ▼
             QoS Prediction
                    │
                    ▼
             Good / Fair / Poor

                    +
                    │
                    ▼
           KPI Labeling Strategy
                    │
                    ▼
       Normal / Warning / Critical

                    │
                    ▼
          Streamlit AIOps Dashboard
```

# Project Results

The project demonstrates an end-to-end telecom AIOps pipeline capable of:

- Monitoring telecom KPIs
- Detecting network anomalies
- Learning temporal network patterns
- Forecasting future throughput
- Identifying probable causes of anomalies
- Recommending corrective actions
- Predicting QoS/QoE
- Generating network health labels
- Providing interactive network monitoring through Streamlit

## Author
**Vibhuti Chaddha**
