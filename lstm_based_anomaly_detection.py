import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
print("Loading dataset...")
df = pd.read_csv("telecom_kpi.csv")
print("\nDataset Loaded Successfully!")
print(df.head())
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)
print("\nDataset Shape:", df.shape)
features = ["RSRP", "SINR","Latency","Throughput","Packet_Loss","Connected_Users"]
X = df[features].values
print("\nSelected Features:")
print(features)
print("\nNormalizing Features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Normalization Completed!")
SEQUENCE_LENGTH = 20
print(f"\nCreating Sliding Windows ({SEQUENCE_LENGTH} time steps)...")
X_sequences = []
y_sequences = []
for i in range(len(X_scaled) - SEQUENCE_LENGTH):
    X_sequences.append(X_scaled[i:i + SEQUENCE_LENGTH])
    y_sequences.append(X_scaled[i + SEQUENCE_LENGTH])
X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)
print("\nSliding Windows Created Successfully!")
print("Input Shape :", X_sequences.shape)
print("Target Shape:", y_sequences.shape)
class TelecomDataset(Dataset):
    def __init__(self, sequences, targets):
        self.X = torch.FloatTensor(sequences)
        self.y = torch.FloatTensor(targets)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.y[index]
dataset = TelecomDataset(X_sequences,y_sequences)
print("\nDataset Created!")
print("Total Samples:", len(dataset))
train_loader = DataLoader(dataset,batch_size=64,shuffle=True)
print("DataLoader Ready!")
for batch_X, batch_y in train_loader:
    print("\nOne Batch Shape")
    print("Input :", batch_X.shape)
    print("Target:", batch_y.shape)
    break
print("\nBuilding LSTM Model")
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, input_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        output = output[:, -1, :]
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction
INPUT_SIZE = len(features)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 20
model = LSTMModel(input_size=INPUT_SIZE,hidden_size=HIDDEN_SIZE,num_layers=NUM_LAYERS)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
print("\nStarting Training...\n")
training_losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    model.train()
    for sequences, targets in train_loader:
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    average_loss = epoch_loss / len(train_loader)
    training_losses.append(average_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}]"f"Loss: {average_loss:.6f}")
print("\nTraining Completed!")
import os
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(),"models/lstm_model.pth")
print("\nModel Saved Successfully!")
print("Location: models/lstm_model.pth")
plt.figure(figsize=(10,5))
plt.plot(range(1, EPOCHS + 1),training_losses,marker="o")
plt.title("LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
print("\nStarting Anomaly Detection")
model.eval()
prediction_errors = []
with torch.no_grad():
    for sequences, targets in train_loader:
        predictions = model(sequences)
        batch_error = torch.mean((predictions - targets) ** 2,dim=1)
        prediction_errors.extend(batch_error.numpy())
prediction_errors = np.array(prediction_errors)
print("Prediction Completed!")
threshold = np.percentile(prediction_errors,95)
print(f"\nAnomaly Threshold : {threshold:.6f}")
results = df.iloc[SEQUENCE_LENGTH:].copy()
results = results.iloc[:len(prediction_errors)].copy()
results["Prediction_Error"] = prediction_errors
results["Anomaly"] = (results["Prediction_Error"] > threshold)
print("\nAnomaly Counts")
print(results["Anomaly"].value_counts())
results.to_csv("lstm_detected_anomalies.csv",index=False)
print("\nCSV Saved Successfully!")
anomalies = results[results["Anomaly"] == True]
anomalies.to_csv("detected_anomalies_only.csv",index=False)
print("Detected Anomalies Saved!")
plt.figure(figsize=(15,5))
plt.plot(results["Timestamp"],results["Throughput"],label="Throughput")
plt.scatter(anomalies["Timestamp"],anomalies["Throughput"],color="red",s=20,label="Anomaly")
plt.title("Throughput Anomalies")
plt.xlabel("Time")
plt.ylabel("Throughput")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(15,5))
plt.plot(results["Timestamp"],results["Latency"],label="Latency")
plt.scatter(anomalies["Timestamp"],anomalies["Latency"],color="red",s=20,label="Anomaly")
plt.title("Latency Anomalies")
plt.xlabel("Time")
plt.ylabel("Latency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(15,5))
plt.plot(results["Timestamp"],results["SINR"],label="SINR")
plt.scatter(anomalies["Timestamp"],anomalies["SINR"],color="red",s=20,label="Anomaly")
plt.title("SINR Anomalies")
plt.xlabel("Time")
plt.ylabel("SINR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(15,5))
plt.plot(results["Timestamp"],results["RSRP"],label="RSRP")
plt.scatter(anomalies["Timestamp"],anomalies["RSRP"],color="red",s=20,label="Anomaly")
plt.title("RSRP Anomalies")
plt.xlabel("Time")
plt.ylabel("RSRP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(15,5))
plt.plot(results["Timestamp"],results["Packet_Loss"],label="Packet Loss")
plt.scatter(anomalies["Timestamp"],anomalies["Packet_Loss"],color="red",s=20,label="Anomaly")
plt.title("Packet Loss Anomalies")
plt.xlabel("Time")
plt.ylabel("Packet Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(15,5))
plt.plot(results["Timestamp"],results["Connected_Users"],label="Connected Users")
plt.scatter(anomalies["Timestamp"],anomalies["Connected_Users"],color="red",s=20,label="Anomaly")
plt.title("Connected Users Anomalies")
plt.xlabel("Time")
plt.ylabel("Connected Users")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print("Total Records :", len(results))
print("Normal Records :", (~results["Anomaly"]).sum())
print("Anomalies :", results["Anomaly"].sum())
print("Threshold :", threshold)