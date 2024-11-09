import streamlit as st
import pandas as pd

st.title('🎁 Machine Learning App')

st.info("This app builds a ML model!")

with st.expander('DATA'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/Reehan09Sarmah/data/refs/heads/main/penguins_cleaned.csv')
  df

  st.write('**X ~ Input**')
  X = df.drop('species', axis=1)
  X

  st.write('**Y ~ Output**')
  Y = df.species
  Y

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Data Preparations
with st.sidebar: # To let users create a data point of their own
  st.header('Input Features')
  # "island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
  island = st.selectbox('Island',('Torgersen', 'Biscoe', 'Dream'))
  gender = st.selectbox('Gender', ('male', 'female'))
  bill_length_mm = st.slider('Bill Length(mm)', min_value=32.1, max_value=59.6, value=43.9)
  
