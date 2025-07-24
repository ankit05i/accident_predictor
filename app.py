import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.title("🚦 Traffic Accident Severity Prediction")
st.write(
    "Upload your dataset and select a target column for prediction using Random Forest."
)

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🗂️ Dataset Preview", df.head())

    # Select target column
    target_col = st.selectbox(
        "🎯 Select the target column (what you want to predict):", df.columns
    )

    # Check if user selected target column
    if target_col:
        # Show distribution of target column
        st.subheader(f"📊 Distribution of {target_col}")
        fig, ax = plt.subplots()
        sns.countplot(x=target_col, data=df, ax=ax)
        st.pyplot(fig)

        # Preprocessing
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Convert target to string (for classification)
        y = y.astype(str)

        # Remove columns in X with mixed types (object columns that are not all strings)
        for col in X.select_dtypes(include=['object']).columns:
            if not X[col].map(type).eq(str).all():
                st.warning(f"Column '{col}' has mixed types and will be dropped.")
                X = X.drop(columns=[col])

        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)

        # Ensure all X columns are numeric
        X = X.apply(pd.to_numeric, errors='coerce')

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        import numpy as np
        # Handle NaNs in X_train and y_train BEFORE fitting the model
        if y_train.isnull().sum() > 0:
            y_train = y_train.fillna(y_train.mode()[0])
        nan_rows = X_train.isnull().any(axis=1) | y_train.isnull()
        if nan_rows.any():
            X_train = X_train.loc[~nan_rows]
            y_train = y_train.loc[~nan_rows]

        # Also handle NaNs in X_test (for prediction)
        if X_test.isnull().any().any():
            X_test = X_test.fillna(0)

        # Train Random Forest
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        st.subheader("✅ Model Evaluation")
        st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Plot Feature Importance
        st.subheader("🔥 Top Feature Importances")
        importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importances.nlargest(10)
        fig2, ax2 = plt.subplots()
        top_features.plot(kind="barh", ax=ax2)
        st.pyplot(fig2)


