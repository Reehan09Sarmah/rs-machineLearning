import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🎁 Machine Learning App')

st.info("This app builds a ML model!")

with st.expander('DATA'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/Reehan09Sarmah/data/refs/heads/main/penguins_cleaned.csv')
  df

  st.write('**X ~ Input**')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('**Y ~ Output**')
  y_raw = df.species
  y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

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

  input_penguin = pd.DataFrame(data, index=[0])
  input_penguin_data = pd.concat([input_penguin, X_raw], axis=0)

# Input Features
with st.expander('Input Features'):
  st.write('**Input Penguin Features**')
  input_penguin
  st.write('**Combined Data**')
  input_penguin_data
  

# Data preparation
# Encode string categorical input data(X) using One Hot Encoding
encode = ['island', 'sex']
encoded_penguin_data = pd.get_dummies(input_penguin_data, prefix=encode, columns=encode)
encoded_penguin_data_Xtrain = encoded_penguin_data[1:] # for the model to train on
encoded_input_penguin = encoded_penguin_data[:1] # the data row we created

# Encode Output Data(Y)
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2} # assigning numerical value to the categories

def target_encode(val):
  return target_mapper[val]

encoded_y = y_raw.apply(target_encode)  #applies above funtion to each value of Y_raw and returns new encoded data.



with st.expander('Data Preparation'):
  st.write('**Encoded X(Penguin Features)**')
  encoded_penguin_data_Xtrain
  st.write('**Encoded y**')
  encoded_y

# Model Training - Use encoded ones.
clf = RandomForestClassifier()
clf.fit(encoded_penguin_data_Xtrain, encoded_y)



# Prediction of Species
with st.expander('Prediction'):
  # predictions
  prediction = clf.predict(encoded_input_penguin)
  prediction_proba = clf.predict_proba(encoded_input_penguin)

  df_prediction_probs = pd.DataFrame(prediction_proba)
  df_prediction_probs.rename(columns={0:'Adelie',
                                     1: 'Chinstrap',
                                     2: 'Gentoo'})

  df_prediction_probs

  keys = list(target_mapper.keys())
  species = None
  for key in keys:
    if(prediction == target_mapper[key]):
      species = key
      
  st.write('**Input Data**')
  input_penguin
  st.write('**Prediction**')
  species
  

    



  
  
