import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("appointments.csv")


df = df.drop(['PatientId', 'AppointmentID'], axis=1)


df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])  
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])  


df['DaysUntilAppointment'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df = df[df['DaysUntilAppointment'] >= 0]

label_columns = ['Gender', 'Neighbourhood', 'No-show']
for col in label_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

selected_features = ['Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes',
                     'Alcoholism', 'Handcap', 'SMS_received', 'DaysUntilAppointment']

X = df[selected_features]
y = df['No-show']  


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


predictions = rf_model.predict(X_val)
predicted_probs = rf_model.predict_proba(X_val)[:, 1]  


def recommend_intervention(probability):
    if probability > 0.7:
        return "Call Patient"  
    elif probability > 0.4:
        return "Send SMS Reminder"  
    else:
        return "No Action Needed"  


intervention_plan = []
for p in predicted_probs:
    intervention_plan.append(recommend_intervention(p))


results_df = X_val.copy()
results_df['Actual'] = y_val.values
results_df['Predicted Probability'] = predicted_probs
results_df['Intervention'] = intervention_plan


print(confusion_matrix(y_val, predictions))
print(classification_report(y_val, predictions))
print(results_df.head(10))
