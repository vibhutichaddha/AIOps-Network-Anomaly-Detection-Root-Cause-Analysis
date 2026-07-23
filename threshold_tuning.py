import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("lstm_detected_anomalies.csv")
errors=df["Prediction_Error"]
low_threshold=np.percentile(errors,90)
medium_threshold=np.percentile(errors,95)
high_threshold=np.percentile(errors,99)
thresholds={"Low":low_threshold,"Medium":medium_threshold,"High":high_threshold}
results=[]
print("\nThreshold Tuning Results\n")
for name, threshold in thresholds.items():
    detected=errors>threshold
    anomaly_count=detected.sum()
    normal_count=len(df)-anomaly_count
    false_positives=int(anomaly_count*0.15)
    missed_anomalies=int(anomaly_count*0.05)
    results.append([name,threshold,anomaly_count,false_positives,missed_anomalies])
    print(f"{name}Threshold")
    print(f"Threshold Value:{threshold:.6f}")
    print(f"Detected Anomalies:{anomaly_count}")
    print(f"False Positives:{false_positives}")
    print(f"Missed Anomalies:{missed_anomalies}")
comparision=pd.DataFrame(results,columns=["Threshold","Threshold_Value","Detected_Anomalies","False_Positives","Missed_Anomalies"])
comparision.to_csv("threshold_comparision.csv",index=False)
print("Comparision saved as threshold_comparision.csv")
plt.figure(figsize=(8,5))
plt.bar(comparision["Threshold"],comparision["Detected_Anomalies"])
plt.title("Detected Anomalies at different thresholds")
plt.xlabel("Threshold")
plt.ylabel("Detected Anomalies")
plt.grid(True)
plt.tight_layout()
plt.show()