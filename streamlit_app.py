import streamlit as st
import pandas as pd

st.title('🎁 Machine Learning App')

st.info("This app builds a ML model!")

with st.expander('DATA'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/Reehan09Sarmah/data/refs/heads/main/penguins_cleaned.csv')
  df
