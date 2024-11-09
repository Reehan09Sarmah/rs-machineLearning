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

# To let users create a data point of their own and add to the existing data
with st.sidebar: 
  st.header('Input Features')
  # "island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
  island = st.selectbox('Island',('Torgersen', 'Biscoe', 'Dream'))
  bill_length_mm = st.slider('Bill Length(mm)', min_value=32.1, max_value=59.6, value=43.9)
  bill_depth_mm = st.slider('Bill Depth(mm)', min_value=13.1, max_value=21.5, value=17.2)
  flipper_length_mm = st.slider('Flipper Length(mm)', min_value=172.0, max_value=231.0, value=201.0)
  body_mass_g = st.slider('Body Mass(g)', min_value=2700, max_value=6300, value=4207)
  gender = st.selectbox('Gender', ('male', 'female'))
  
  # create a dataframe
  data = {'island': island,
         'bill_length_mm': bill_length_mm,
         'bill_depth_mm': bill_depth_mm,
         'flipper_length_mm': flipper_length_mm,
         'body_mass_g': body_mass_g,
         'sex': gender}

  input_df = pd.DataFrame(data, index=[0])
  input_penguin_data = pd.concat([input_df, X], axis=0)

with st.expander('Input Features'):
  st.write('**Input Penguin Features**')
  input_df
  st.write('**Combined Data**')
  input_penguin_data

# Encode string data into numerical
df_encoded = pd.get_dummies(input_penguin_data, prefix=encode, columns=['island', 'sex'])
df_encoded[:1]
  
  
