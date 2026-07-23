import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
st.set_page_config(page_title="Telecom_AIOps_Dashboard",layout="wide")
df=pd.read_csv("telecom_kpi.csv")
df["Timestamp"]=pd.to_datetime(df["Timestamp"])
try:
    anomalies=pd.read_csv("detected_anomalies.csv")
except:
    anomalies=pd.DataFrame()
try:
    rca=pd.read_csv("root_cause_analysis.csv")
except:
    rcs=pd.DataFrame()
try:
    labels=pd.read_csv("telecom_lpi_labeled.csv")
except:
    labels=pd.DataFrame()
st.sidebar.title("Telecom AIOps")
cell=st.sidebar.selectbox("Select Cell",sorted(df["Cell_ID"].unique()))
filtered=df[df["Cell_ID"]==cell]
st.title("Telecom Network AIOps Dasboard")
st.header("Live KPI Trends")
metric=st.selectbox("Select KPI",["RSRP","SINR","Latency","Throughput","Packet_Loss","Connected_Users"])
fig=px.line(filtered,x="Timestamp",y=metric,title=f"{metric}Trend")
st.plotly_chart(fig,use_container_width=True)
st.header("Detected Anomalies")
if not anomalies.empty:
    st.dataframe(anomalies.head(20))
else:
    st.info("No anomaly file found")
st.header("Throughput Forecast")
forecast=filtered.tail(288)
forecast_fig=px.line(forecast,x="Timestamp",y="Throughput",title="Next 24 Hours Forecast")
st.plotly_chart(forecast_fig,use_container_width=True)
st.header("QoS/QoE Prediction")
try:
    model=joblib.load("qos_prediction_model.pkl")
    latest=filtered[["RSRP","SINR","Latency","Packet_Loss","Throughput"]].tail(1)
    prediction=model.predict(latest)[0]
    label_map={0:"Fair",1:"Good",2:"Poor"}
    st.success(f"Predicted Network Quality:{label_map[prediction]}")
except:
    st.warning("QoS model not available")
st.header("Root Cause Analysis")
if not rca.empty:
    st.dataframe(rca[["Observation","Possible_Root_Cause","Recommended_Action"]].head(15))
else:
    st.info("RCA file not available")
st.header("KPI Threshold Alerts")
latest=filtered.iloc[-1]
alerts=[]
if latest["RSRP"]<-105:
    alerts.append("Critical RSRP")
if latest["SINR"]<10:
    alerts.append("Critical SINR")
if latest["Latency"]>50:
    alerts.append("Critical Latency")
if latest["Packet_Loss"]>3:
    alerts.append("Critical Packet Loss")
if alerts:
    for alert in alerts:
        st.error(alert)
else:
    st.success("No Active Alert")