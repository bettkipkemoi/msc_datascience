import streamlit as st
import pandas as pd
import janitor
import plotly as plt
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Convenience Sales Dashboard",
    page_icon="📊",
    layout="wide"
)
st.title("Convenience Sales Dashboard")

# load data
#@st.cache_data
df = pd.read_csv("bettkipkemoi/msc_datascience/main/dsc_808-bi_data_analytics/interactive-dashboard/202445.csv", engine='python', encoding='unicode-escape')
df = df.clean_names()
print(df.columns)
#remove pounds sign from total sales, avg_basket_size_prod_sales_, and aip columns and convert to numeric
df['total_sales_sales_inc_vat_inc_di_'] = df['total_sales_sales_inc_vat_inc_di_'].replace('[£,]', '', regex=True).astype(float)
df['avg_basket_size_prod_sales_'] = df['avg_basket_size_prod_sales_'].replace('[£,]', '', regex=True).astype(float)
df['aip'] = df['aip'].replace('[£,]', '', regex=True).astype(float)

st.sidebar.header("Filter Options")
# create a sidebar filter for region
region_filter = st.sidebar.multiselect(
    "Select Region:",
    options=df["region_"].unique(),
    default=df["region_"].unique()
)

# Apply the filter to the dataframe
filtered_df = df[df["region_"].isin(region_filter)]

st.markdown("### Region Selected: {region_filter}".format(region_filter=region_filter))

col1, col2 = st.columns(2)
with col1:
    total_sales = int(filtered_df["total_sales_sales_inc_vat_inc_di_"].sum())
    st.metric("Total Sales (Inc. VAT & DI)", f"£{total_sales:,}")
with col2:
    distinct_customers = int(filtered_df["distinct_customers"].sum())
    st.metric("Distinct Customers", f"{distinct_customers:,}")


tab1, tab2, tab3 = st.tabs(["Sales by Region", "Customers by Region", "Pie Chart: Sales Distribution"])

with tab1:
    st.subheader("Sales by Region")
    fig1 = px.bar(
        filtered_df,
        x='region_',
        y='total_sales_sales_inc_vat_inc_di_',
        labels={"region_": "Region", "total_sales_sales_inc_vat_inc_di_": "Total Sales (Inc. VAT & DI)"},
        title="Total Sales by Region",
        color='region_',
        hover_data='region_',
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader("Customers by Region")
    fig2 = px.bar(
        filtered_df,
        x='region_',
        y='distinct_customers',
        labels={"region_": "Region", "distinct_customers": "Distinct Customers"},
        title="Distinct Customers by Region",
        color='region_',
        hover_data='region_',
    )
    st.plotly_chart(fig2, use_container_width=True)
    
with tab3:
    st.subheader("Pie Chart: Sales Distribution by Region")
    pie_chart_sales = px.pie(
        filtered_df,
        names='region_',
        values='total_sales_sales_inc_vat_inc_di_',
        title='Sales Distribution by Region',
        hole=0.4
    )
    st.plotly_chart(pie_chart_sales, use_container_width=True)

st.markdown("---")
st.markdown("Developed by Bett Kipkemoi")