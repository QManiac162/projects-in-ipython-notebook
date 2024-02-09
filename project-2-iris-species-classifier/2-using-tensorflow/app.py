import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

# Plotting libraries
import matplotlib.pyplot as plt


# SKLearn libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

# Data file path
FILE_PATH = 'iris_species.csv'

# Dataframe from csv file
iris_data = pd.read_csv(FILE_PATH, header=0)

# scaler will be used to scale user input.
def get_scaler():
    # Clean data
    X = iris_data.iloc[:, :4]
    y = np.zeros(shape=(X.shape[0], 3))

    for i, val in enumerate(iris_data['variety']):
        if val=='Virginica':
            y[i,:] = np.array([1, 0, 0])
        elif val=='Versicolor':
            y[i,:] = np.array([0, 1, 0])
        elif val=='Setosa':
            y[i,:] = np.array([0, 0, 1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100)

    # Scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

scaler = get_scaler()

# Load model
model = keras.models.load_model("iris_model.keras")

# App title and description
st.title('Iris Flower Classifier')
st.markdown("""Predict the species of an Iris flower using sepal and petal measurements.""")

# Define components for the sidebar
st.sidebar.header('Input Features')
sepal_length = st.sidebar.slider(
    label='Sepal Length',
    min_value=iris_data['sepal.length'].min(),
    max_value=iris_data['sepal.length'].max(),
    value=round(iris_data['sepal.length'].mean(), 1),
    step=0.1)
sepal_width = st.sidebar.slider(
    label='Sepal Width',
    min_value=iris_data['sepal.width'].min(),
    max_value=iris_data['sepal.width'].max(),
    value=round(iris_data['sepal.width'].mean(), 1),
    step=0.1)
petal_length = st.sidebar.slider(
    label='Petal Length',
    min_value=iris_data['petal.length'].min(),
    max_value=iris_data['petal.length'].max(),
    value=round(iris_data['petal.length'].mean(), 1),
    step=0.1)
petal_width = st.sidebar.slider(
    label='Petal Width',
    min_value=iris_data['petal.width'].min(),
    max_value=iris_data['petal.width'].max(),
    value=round(iris_data['petal.width'].mean(), 1),
    step=0.1)

# Scale the user inputs
X_scaled = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])

# Run input through the model
y_pred = model.predict(X_scaled)
df_pred = pd.DataFrame({
    'Species': ['Virginica', 'Versicolor', 'Setosa'], 'Confidence': y_pred.flatten()
})

# Define the prediction bar chart.
fig = px.bar(
    df_pred, 
    x='Species', 
    y='Confidence',
    width=350, 
    height=350, 
    color='Species',
    color_discrete_sequence =['#00CC96', '#EF553B', '#636EFA'])

# Create two columns for the web app.
# Column 1 will be for the predictions.
# Column 2 will be for the PCA.
# Make the second column 20% wider than the first column

col1, col2 = st.columns((1, 1.2))
with col1:
    st.markdown('### Predictions')
    fig

def run_pca():
    # Run PCA
    pca = PCA(2)
    X = iris_data.iloc[:, :4]
    X_pca = pca.fit(X).transform(X)
    df_pca = pd.DataFrame(pca.transform(X))
    df_pca.columns = ['PC1', 'PC2']
    df_pca = pd.concat([df_pca, iris_data['variety']], axis=1)
    
    return pca, df_pca

pca, df_pca = run_pca()
# Create the PCA chart
pca_fig = px.scatter(
    df_pca, 
    x='PC1', 
    y='PC2', 
    color='variety', 
    hover_name='variety', 
    width=500, 
    height=350)

# Retrieve user input
datapoint = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])
# Map the 4-D user input to 2-D using the PCA
datapoint_pca = pca.transform(datapoint)
# Add the user input to the PCA chart
pca_fig.add_trace(go.Scatter(
        x=[datapoint_pca[0, 0]], 
        y=[datapoint_pca[0,1]], 
        mode='markers', 
        marker={'color': 'black', 'size':10}, name='Your Datapoint'))

with col2:
    st.markdown('### Principle Component Analysis')
    pca_fig