import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib 
df=pd.read_csv("telecom_kpi.csv")
df["Timestamp"]=pd.to_datetime(df["Timestamp"])
df=df.sort_values("Timestamp")
features=["RSRP","SINR","Latency","Throughput","Packet_Loss","Connected_Users"]
X=df[features]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
model=IsolationForest(n_estimators=100,contamination=0.02,random_state=42)
model.fit(X_scaled)
print("Isolation Forest trained successfully")
df["Anomaly"]=model.predict(X_scaled)
df["Anomaly"]=df["Anomaly"].map({-1:0,-1:1})
print("\nAnomaly Counts")
print(df["Anomaly"].value_counts())
joblib.dump(model,"isolation_forest_model.pkl")
print("\nModel saved as isolation_forest_model.pkl")
joblib.dump(scaler,"scaler.pkl")
print("\nScaler saved as scaler.pkl")
anomalies = df[df["Anomaly"] == 1]
print("\nDetected Anomalies")
print(anomalies.head())
anomalies.to_csv("detected_anomalies.csv",index=False)
print("\nDetected anomalies saved!")
def plot_anomaly(kpi):
    plt.figure(figsize=(15,5))
    plt.plot(df["Timestamp"],df[kpi],label=kpi)
    plt.scatter(anomalies["Timestamp"],anomalies[kpi],color="red",s=20,label="Anomaly")
    plt.title(f"{kpi} with Detected Anomalies")
    plt.xlabel("Timestamp")
    plt.ylabel(kpi)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
plot_anomaly("Throughput")
plot_anomaly("Latency")
plot_anomaly("SINR")
plot_anomaly("RSRP")
plot_anomaly("Packet_Loss")
plot_anomaly("Connected_Users")
print("\nSUMMARY")
print("Total Records :", len(df))
print("Normal Records :", (df["Anomaly"] == 0).sum())
print("Anomalies :", (df["Anomaly"] == 1).sum())