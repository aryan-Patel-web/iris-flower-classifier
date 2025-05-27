import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Important decorator for caching
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

# Load dataset
df, target_names = load_data()

# Train the model
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df["species"])

# Streamlit UI
st.title("🌸 Iris Flower Classification App")

st.sidebar.header("🔹 Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

# Debugging: Print column names to check
st.write("Column Names:", df.columns.tolist())

# Prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
prediction_class = target_names[prediction[0]]

# Display Prediction
st.subheader("📌 Prediction")
st.write(f"🌿 The predicted species type is: **{prediction_class}**")

st.write("This model uses a Random Forest Classifier trained on the Iris dataset to predict the species based on input features.")

# Display dataset for reference
st.subheader("📊 Iris Dataset Sample")
st.dataframe(df.head())
