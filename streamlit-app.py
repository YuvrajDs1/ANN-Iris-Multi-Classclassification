import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

model = load_model("iris_classifier.h5")

st.header('Iris Classification using ANN')
sepal_len = st.slider("sepal length (cm)", 4.3,7.9,0.1)
sepal_wid = st.slider("sepal width (cm)", 2.0, 4.4, 0.1)
petal_len = st.slider("petal length (cm)", 1.0, 6.9, 0.1)
petal_wid = st.slider("petal width (cm)", 0.1, 2.5, 0.1)


input_data = {
    'sepal length (cm)': sepal_len, 
    'sepal width (cm)':sepal_wid, 
    'petal length (cm)':petal_len,
    'petal width (cm)': petal_wid
}

input_data_df = pd.DataFrame([input_data]).astype(np.float64)

prediction = model.predict(input_data_df)

prediction_cat = np.argmax(prediction)
classes = ['Setosa', 'Versicolor', 'Virginica']

st.write('Prediction:', classes[prediction_cat])

probs = prediction.flatten()

import altair as alt

df = pd.DataFrame({
    'Class': classes,
    'Probability': probs
})

chart = alt.Chart(df).mark_bar().encode(
    x='Class',
    y='Probability',
    color='Class'
).properties(title='Prediction Probabilities')

st.altair_chart(chart, use_container_width=True)