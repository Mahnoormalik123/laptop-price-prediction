import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Title of the app
st.title('Laptop Price Prediction with Streamlit')

# Upload dataset
uploaded_file = st.file_uploader("laptop_data_cleaned.csv", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())
    
    # Display column names and types
    st.write("Column Names:", data.columns)
    st.write(data.dtypes)

    # Encode categorical columns
    categorical_columns = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'Os']  # Update with your actual categorical columns
    le = LabelEncoder()
    for col in categorical_columns:
        if col in data.columns:
            data[col] = le.fit_transform(data[col])
    
    # Perform one-hot encoding if needed
    # data_encoded = pd.get_dummies(data, drop_first=True)
    
    # Assuming 'Price' is the target column
    target_column = 'Price'
    if target_column in data.columns:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # Ensure there are no NaN values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Define models
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=0),
            "Support Vector Regressor": SVR(),
            "Random Forest Regressor": RandomForestRegressor(random_state=0)
        }

        # Model selection dropdown
        model_name = st.selectbox("Select a model", list(models.keys()))

        # Train and evaluate the selected model
        if model_name:
            model = models[model_name]
            st.write(f"Training {model_name}...")
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                st.write(f"{model_name} Metrics:")
                st.write(f"  Mean Squared Error: {mse:.4f}")
                st.write(f"  Root Mean Squared Error: {rmse:.4f}")
                st.write(f"  R² Score: {r2:.4f}")
            except Exception as e:
                st.write(f"Failed to process {model_name}: {e}")
    else:
        st.write(f"Target column '{target_column}' not found in the dataset.")
