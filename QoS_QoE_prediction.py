import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,f1_score,recall_score,classification_report,confusion_matrix)
import joblib
df=pd.read_csv("telecom_kpi.csv")
if "QoS_Label" not in df.columns:
    def qos(row):
        if(row["RSRP"]>-90 and row["SINR"]>20 and row["Latency"]<20 and row["Packet_Loss"]<1 and row["Throughput"]>80):
            return "Good"
        elif(row["RSRP"]>-105 and row["SINR"]>10 and row["Latency"]<50 and row["Packet_Loss"]<3 and row["Throughput"]>40):
            return "Fair"
        else:
            return "Poor"
    df["QoS_Label"]=df.apply(qos,axis=1)
X=df[["RSRP","SINR","Latency","Packet_Loss","Throughput"]]
y=df["QoS_Label"]
encoder=LabelEncoder()
y=encoder.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
predictions=model.predict(X_test)
accuracy=accuracy_score(y_test,predictions)
precision=precision_score(y_test,predictions,average="weighted")
f1=f1_score(y_test,predictions,average="weighted")
recall=recall_score(y_test,predictions,average="weighted")
print("\nQoS Prediction\n")
print(f"Accuracy:{accuracy:.4f}")
print(f"Precision:{precision:.4f}")
print(f"F1 Score:{f1:.4f}")
print(f"Recall:{recall:.4f}")
print("\nClassification Report\n")
print(classification_report(y_test,predictions,target_names=encoder.classes_))
print("\nConfusion Matrix\n")
print(confusion_matrix(y_test,predictions))
joblib.dump(model,"qos_prediction_model.pkl")
print("\nModel Saved Successfully!")
print("File: qos_prediction_model.pkl")