# import necessary libraries
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="Monthly Volume of Airline Passengers in 90 Countries between 2010-2018", layout="wide")
st.title("Global Flight Data Visualization")

url = 'https://raw.githubusercontent.com/bettkipkemoi/msc_datascience/refs/heads/main/dsc_808-bi_data_analytics/interactive-dashboard/monthly_vol_of_airline_pass_in_90_countries_2010_2018.csv'
geo_url = 'https://github.com/bettkipkemoi/msc_datascience/tree/main/dsc_808-bi_data_analytics/interactive-dashboard/ne_110m_admin_0_countries'
@st.cache_data
def load_data():
    # load flight data from github repository
    df = pd.read_csv(url)
    gdf = gpd.read_file(geo_url)

    # ensure year and month are integers
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    # fill nas with 0 for Total_OS for plotting
    df['Total_OS'] = df['Total_OS'].fillna(0)
    return df


#@st.cache_data
# load countries centroids from the shapefiles in geo_url using geopandas



try:
    # --- Load Data ---
    df = load_data()
    centroids = gdf.centroid
    for index, point in enumerate(centroids):
        print(f"Country {gdf.iloc[index]['NAME']}: Lon={point.x}, Lat={point.y}")

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Data")
    
    # 2. Pick a Country with a selectbox
    country = st.selectbox(label="Select a country", options=[""] + list(df["ISO3"].unique()))

    # 3. Pick a Year and Month with a selectbox
    selected_month = st.selectbox("Select Month", sorted(df['Month'].unique()))
    selected_year = st.selectbox("Select Year", sorted(df['Year'].unique()))
    selected_country = country if country else df['ISO3'].iloc[0]
    # --- Data Processing ---
    # Filter the dataframe based on selection
    filtered_df = df[(df['ISO3'] == selected_country) & (df['Month'] == selected_month)]

    if not filtered_df.empty:
        row = filtered_df.iloc[0]
        passengers = row['Total_OS']
        
        # Display Metrics
        col1, col2 = st.columns(2)
        col1.metric(f"Country: {selected_country}", f"Month: {selected_month}")
        col2.metric("Total Operational Seats (Total_OS)", f"{passengers:,.0f}")

        # --- Folium Map Visualization ---
        st.subheader(f"Map: Passengers in {selected_country} (Month {selected_month})")
        
        # Get coordinates for the selected country
        location = coords_dict.get(selected_country, [0, 0])
        
        # Initialize Map
        m = folium.Map(location=location, zoom_start=5, tiles="CartoDB positron")

        # Add a CircleMarker representing the passenger volume
        # We scale the radius based on passenger count
        radius = (passengers ** 0.5) / 10 # Square root scaling for better visual growth
        
        folium.CircleMarker(
            location=location,
            radius=max(radius, 5), # Minimum visible radius
            popup=f"{selected_country}: {passengers:,.0f} seats",
            color="crimson",
            fill=True,
            fill_color="crimson",
            fill_opacity=0.6
        ).add_to(m)

        # Render Map in Streamlit
        st_folium(m, width=1000, height=500)
        
    else:
        st.warning("No data found for the selected Country and Month.")

except Exception as e:
    st.error(f"Error: {e}")


# plot a time series of total operational seats over the years for the selected country
st.subheader(f"Time Series: Total Operational Seats in {selected_country} Over Years")
time_series_df = df[df['ISO3'] == selected_country].groupby(['Year', 'Month']).agg({'Total_OS': 'sum'}).reset_index()
time_series_df['Date'] = pd.to_datetime(time_series_df[['Year', 'Month']].assign(DAY=1))
fig = px.line(time_series_df, x='Date', y='Total_OS', title=f'Time Series of Total Operational Seats in {selected_country}',
              labels={'Total_OS': 'Total Operational Seats', 'Date': 'Date'})
st.plotly_chart(fig, use_container_width=True)

# plot a bar chart of total operational seats by country for the selected month and year
st.subheader(f"Bar Chart: Total Operational Seats by Country in {selected_month}/{selected_year}")
bar_chart_df = df[(df['Year'] == selected_year) & (df['Month'] == selected_month)]
fig2 = px.bar(bar_chart_df, x='ISO3', y='Total_OS', title=f'Total Operational Seats by Country in {selected_month}/{selected_year}',
              labels={'Total_OS': 'Total Operational Seats', 'ISO3': 'Country'})
st.plotly_chart(fig2, use_container_width=True)