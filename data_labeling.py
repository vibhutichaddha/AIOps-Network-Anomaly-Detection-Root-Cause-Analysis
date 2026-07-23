import pandas as pd 
df=pd.read_csv("telecom_kpi.csv")
def classify_kpi(row):
    normal=0
    warning=0
    critical=0
    if row["RSRP"]>-90:
        normal+=1
    elif row["RSRP"]<-105:
        critical+=1
    else:
        warning+=1
    if row["SINR"]>20:
        normal+=1
    elif row["SINR"]>=10:
        critical+=1
    else:
        warning+=1
    if row["Latency"]<20:
        normal+=1
    elif row["Latency"]<=50:
        critical+=1
    else:
        warning+=1
    if row["Packet_Loss"]<1:
        normal+=1
    elif row["Packet_Loss"]<=3:
        critical+=1
    else:
        warning+=1 
    if critical>=1:
        return "Critical"
    elif warning>=2:
        return "Warning"
    else:
        return "Normal"
df["KPI_Label"]=df.apply(classify_kpi,axis=1)
df.to_csv("telecom_kpi_labeled.csv",index=False)
print("Labeling Completed!\n")
print(df["KPI_Label"].value_counts())