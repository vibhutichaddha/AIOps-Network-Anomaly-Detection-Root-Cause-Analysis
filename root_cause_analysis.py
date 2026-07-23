import pandas as pd
df=pd.read_csv("lstm_detected_anomalies.csv")
anomalies=df[df["Anomaly"]==True].copy()
print("Total detected anomalies:",len(anomalies))
def identify_root_cause(row):
    observations=[]
    causes=[]
    actions=[]
    if row["RSRP"]<-110:
        observations.append("Low RSRP")
        causes.append("Poor radio coverage")
        actions.append("Optimize antenna tilt")
    if row["SINR"]<10:
        observations.append("Low SINR")
        causes.append("High Interferene")
        actions.append("Review neighboring cell configuration")
    if row["Latency"]>80:
        observations.append("High Latency")
        causes.append("Core network congestion")
        actions.append("Check UPF utilization")
    if row["Packet_Loss"]>3:
        observations.append("High Packet Loss")
        causes.append("Backhaul issues")
        actions.append("Inspect transport network")
    if row["Throughput"]<30:
        observations.append("Low Throughput")
        causes.append("Heavy Traffic Load")
        actions.append("Enable Load Balancing")
    if row["Connected_Users"] > 180:
        observations.append("High Connected Users")
        causes.append("Cell overload")
        actions.append("Deploy small cells or rebalance users")
    if len(observations) == 0:
        observations.append("No abnormal KPI")
        causes.append("Unknown")
        actions.append("Further investigation required")
    return pd.Series([", ".join(observations),", ".join(causes),", ".join(actions)])
anomalies[["Observation","Possible_Root_Cause","Recommended_Action"]]=anomalies.apply(identify_root_cause,axis=1)
anomalies.to_csv("root_cause_analysis.csv",index=False)
print("\nSummary:")
print(anomalies["Possible_Root_Cause"].value_counts())