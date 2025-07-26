import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load data
df = pd.read_csv("dataset_traffic_accident_prediction1.csv")
print("Shape:", df.shape)

# 2. Basic cleaning & encode
df.fillna(method='ffill', inplace=True)
cat_cols = df.select_dtypes(['object']).columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 3. EDA plots
plt.figure(figsize=(6,4))
sns.countplot(x='Accident_Severity', data=df)
plt.title("Accident Severity Distribution")
plt.savefig("severity_dist.png")
plt.close()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.close()

# 4. Train/test split
X = df.drop("Accident_Severity", axis=1)
y = df["Accident_Severity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. RandomForest (default params)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
top10 = importances.nlargest(10)
top10.plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.savefig("feature_importance.png")
plt.close()

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Traffic Accident Severity Predictor")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    if st.button("Run Prediction"):
        # Preprocess & train model
        # ...
        st.success("Prediction complete!")
