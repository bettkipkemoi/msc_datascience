# import necessary libraries
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="Monthly Volume of Airline Passengers in 90 Countries between 2010-2018", layout="wide")
st.title("Global Flight Data Visualization")

@st.cache_data
def load_data():