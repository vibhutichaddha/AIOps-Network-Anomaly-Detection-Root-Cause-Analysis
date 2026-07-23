import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error,mean_squared_error,mean_absolute_percentage_error)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
df=pd.read_csv("telecom_kpi.csv")
df["Timestamp"]=pd.to_datetime(df["Timestamp"])
df=df.sort_values("Timestamp")
throughput=df["Throughput"].values.reshape(-1,1)
scaler=MinMaxScaler()
scaled=scaler.fit_transform(throughput)
sequence_length=20
X=[]
y=[]
for i in range(len(scaled)-sequence_length):
    X.append(scaled[i:i+sequence_length])
    y.append(scaled[i+sequence_length])
X=np.array(X)
y=np.array(y)
X=torch.FloatTensor(X)
y=torch.FloatTensor(y)
dataset=TensorDataset(X,y)
loader=DataLoader(dataset,batch_size=64,shuffle=True)
class ForecastLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(input_size=1,hidden_size=64,num_layers=2,batch_first=True)
        self.fc=nn.Linear(64,1)
    def forward(self,x):
        output,_=self.lstm(x)
        output=output[:,-1,:]
        output=self.fc(output)
        return output
model=ForecastLSTM()
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
epochs = 20
losses = []
for epoch in range(epochs):
    total_loss = 0
    for inputs,target in loader:
        optimizer.zero_grad()
        prediction = model(inputs)
        loss = criterion(prediction,target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss/len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs} Loss {avg_loss:.5f}")
torch.save(model.state_dict(),"forecast_lstm.pth")
print("Model Saved")
future_steps = 288
model.eval()
window = scaled[-sequence_length:]
forecast = []
with torch.no_grad():
    for _ in range(future_steps):
        sequence = torch.FloatTensor(window).unsqueeze(0)
        prediction = model(sequence)
        value = prediction.item()
        forecast.append(value)
        window = np.vstack((window[1:], [[value]]))
forecast = scaler.inverse_transform(np.array(forecast).reshape(-1,1))
future_steps_7=2016
window=scaled[-sequence_length:]
forecast7=[]
with torch.no_grad():
    for _ in range(future_steps_7):
         sequence = torch.FloatTensor(window).unsqueeze(0)
         prediction = model(sequence)
         value = prediction.item()
         forecast7.append(value)
         window = np.vstack((window[1:], [[value]]))
forecast7 = scaler.inverse_transform(np.array(forecast7).reshape(-1,1))
actual = throughput[-288:]
predicted = forecast[:288]
mae = mean_absolute_error(actual,predicted)
rmse = np.sqrt(mean_squared_error(actual,predicted))
mape = mean_absolute_percentage_error(actual,predicted)
print()
print("Prediction Accuracy")
print("MAE :",mae)
print("RMSE :",rmse)
print("MAPE :",mape)
plt.figure(figsize=(14,5))
plt.plot(actual,label="Actual Throughput")
plt.plot(predicted,label="Forecast")
plt.title("24 Hour Throughput Forecast")
plt.xlabel("Time Steps")
plt.ylabel("Throughput (Mbps)")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(14,5))
plt.plot(forecast7,label="7-Day Forecast")
plt.title("7-Day Throughput Forecast")
plt.xlabel("Future Time Steps")
plt.ylabel("Throughput (Mbps)")
plt.legend()
plt.grid(True)
plt.show()