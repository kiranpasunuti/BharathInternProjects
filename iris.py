import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
data = pd.read_csv('iris.csv')
data.rename(columns={'Sepal.Length': 'Sepal_Length'}, inplace=True)
data.rename(columns={'Sepal.Width': 'Sepal_Width'}, inplace=True)
data.rename(columns={'Petal.Length': 'Petal_Length'}, inplace=True)
data.rename(columns={'Petal.Width': 'Petal_Width'}, inplace=True)
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=200)
model = LogisticRegression()
model.fit(x_train, y_train)

# Create a Streamlit app
st.title("Iris Species Prediction")

# Input form for user to enter Sepal and Petal details
sepal_length = st.slider("Sepal Length", min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.slider("Sepal Width", min_value=2.0, max_value=4.5, step=0.1)
petal_length = st.slider("Petal Length", min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.slider("Petal Width", min_value=0.1, max_value=2.5, step=0.1)

# Make prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]

# Convert the label back to original species
predicted_species = le.inverse_transform([prediction])[0]

# Display the prediction with styling
st.subheader("Prediction:")
st.markdown(f"<span style='color:green; font-weight:bold'>{predicted_species}</span>", unsafe_allow_html=True)
