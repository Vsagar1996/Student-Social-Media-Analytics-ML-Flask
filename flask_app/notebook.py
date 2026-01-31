import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Students Social Media Addiction.csv")
df.head(20)

categorical_cols = [
    'Gender',
    'Academic_Level',
    'Country',
    'Most_Used_Platform',
    'Relationship_Status'
]

df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

df_encoded['Affects_Academic_Performance'] = (
    df_encoded['Affects_Academic_Performance']
    .map({'Yes': 1, 'No': 0})
)

numerical_cols = [
    'Age',
    'Avg_Daily_Usage_Hours',
    'Mental_Health_Score'
]

scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(
    df_encoded[numerical_cols]
)


# Linear Regression

X_lr = df_encoded.drop(
    ['Student_ID', 'Addicted_Score', 'Affects_Academic_Performance'],
    axis=1
)

y_lr = df_encoded['Addicted_Score']

X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(
    X_lr, y_lr, test_size=0.2, random_state=45
)

lr = LinearRegression()
lr.fit(X_lr_train, y_lr_train)

y_lr_pred = lr.predict(X_lr_test)

# save linear model
pickle.dump(lr, open("linear_model.pkl", "wb"))

# save features
pickle.dump(X_lr.columns, open("columns_lr.pkl", "wb"))


# Logistic Regression

X_log = df_encoded.drop(
    ['Student_ID', 'Addicted_Score', 'Affects_Academic_Performance'],
    axis=1
)

y_log = df_encoded['Affects_Academic_Performance']

X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42
)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_log_train, y_log_train)

y_log_pred = log_reg.predict(X_log_test)

# save the model
pickle.dump(log_reg, open("logistic_model.pkl", "wb"))

# save the features
pickle.dump(X_log.columns, open("columns_log.pkl", "wb"))


# Random Forest

df_encoded['High_Addiction'] = np.where(
    df_encoded['Addicted_Score'] >= 7, 1, 0)

X_rf = df_encoded.drop(
    ['Student_ID', 'Addicted_Score', 'High_Addiction'],
    axis=1
)

y_rf = df_encoded['High_Addiction']

X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=45)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_rf_train, y_rf_train)

y_rf_pred = rf.predict(X_rf_test)

# save the random forest model
pickle.dump(rf, open("rf_model.pkl", "wb"))

# save the features

pickle.dump(X_rf.columns, open("columns_rf.pkl", "wb"))

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))