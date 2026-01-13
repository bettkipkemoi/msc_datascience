import streamlit as st
import pandas as pd
import janitor
import plotly.express as px
# load data
#@st.cache_data
df = pd.read_csv("/Users/bett/downloads/202445.csv", engine='python', encoding='unicode-escape')
df = df.clean_names()
print(df.columns)
st.info('Testing')
st.bar_chart(
    df,
    x="region_",
    y="distinct_customers",
    x_label= "Regions",
    y_label= "Number of Customers"
)

st.title("Interactive Streamlit Pie Chart with Plotly")

# Create the pie chart using plotly.express
fig = px.pie(
    df, 
    values='frequency',  # Column with the values for each slice
    names='region_', # Column with the names for each slice
    title='Sales Distribution',
    hover_data=['frequency'], # Display values in hover tooltip
    hole=0.3 # Optional: Creates a donut chart
)

# Customize the layout (optional)
fig.update_layout(
    margin=dict(l=20, r=20, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True) # use_container_width makes it responsive

# create an interactive plot to select store in a region and see associated data
st.bar_chart(
    df,
    x="region_",
    y="total_sales_sales_inc_vat_inc_di_"
)