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

@st.cache_data
def load_data():
    # load flight data from github repository
    df = pd.read_csv(url)
    # ensure year and month are integers
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    # fill nas with 0 for Total_OS for plotting
    df['Total_OS'] = df['Total_OS'].fillna(0)
    return df


@st.cache_data
# helper to get country cenrtoids
def get_country_coords():

    #access builtin country centroids from geopandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[['iso_a3', 'geometry']]
    world['centroid'] = world.geometry.centroid
    world['lat'] = world.centroid.y
    world['lon'] = world.centroid.x
    # Common centroids for visualization; expanded as needed
    coords = {
        'ALB': [41.1533, 20.1683],
        'USA': [37.0902, -95.7129],
        'GBR': [55.3781, -3.4360],
        # Add more mappings or join with a proper geo-dataset
        'FRA': [46.6034, 1.8883],
        'DEU': [51.1657, 10.4515],
        'IND': [20.5937, 78.9629],
        'CHN': [35.8617, 104.1954],
        'BRA': [-14.2350, -51.9253],
        'ZAF': [-30.5595, 22.9375],
        'AUS': [-25.2744, 133.7751],
        'CAN': [56.1304, -106.3468],
        'JPN': [36.2048, 138.2529],
        'RUS': [61.5240, 105.3188],
        'ESP': [40.4637, -3.7492],
        'ITA': [41.8719, 12.5674],
        'MEX': [23.6345, -102.5528],
        'KOR': [35.9078, 127.7669],
        'TUR': [38.9637, 35.2433],
        'SAU': [23.8859, 45.0792],
        'ARG': [-38.4161, -63.6167],
        # Add more as needed
        'KEN': [-0.0236, 37.9062],
        'NGA': [9.0820, 8.6753],
        'EGY': [26.8206, 30.8025],
        'SWE': [60.1282, 18.6435],
        'NOR': [60.4720, 8.4689],
        'FIN': [61.9241, 25.7482],
        'DNK': [56.2639, 9.5018],
        'POL': [51.9194, 19.1451],
        'GRC': [39.0742, 21.8243],
        'NLD': [52.1326, 5.2913],
        'BEL': [50.5039, 4.4699],
        'CHE': [46.8182, 8.2275],
        'AUT': [47.5162, 14.5501],
        'PRT': [39.3999, -8.2245],
        'IRL': [53.1424, -7.6921],
        'CZE': [49.8175, 15.4730],
        'HUN': [47.1625, 19.5033],
        'ROU': [45.9432, 24.9668],
        'BGR': [42.7339, 25.4858],
        'SVK': [48.6690, 19.6990],
        'HRV': [45.1000, 15.2000],
        'SVN': [46.1512, 14.9955],
        'LTU': [55.1694, 23.8813],
        'LVA': [56.8796, 24.6032],
        'EST': [58.5953, 25.0136],
        'CYP': [35.1264, 33.4299],
        'ISL': [64.9631, -19.0208],
        'ARM': [40.0691, 45.0382],
        'BRB': [13.1939, -59.5432],
        'TUN': [33.8869, 9.5375],
        'MAR': [31.7917, -7.0926],
        'PER': [-9.1899, -75.0152],
        'CHL': [-35.6751, -71.5430],
        'COL': [4.5709, -74.2973],
        'VEN': [6.4238, -66.5897],
        'NZL': [-40.9006, 174.8860],
        'THA': [15.8700, 100.9925],
        'VNM': [14.0583, 108.2772],
        'PHL': [12.8797, 121.7740],
        'IDN': [-0.7893, 113.9213],
        'MYS': [4.2105, 101.9758],
        'SGP': [1.3521, 103.8198],
        'ARE': [23.4241, 53.8478],
        'QAT': [25.3548, 51.1839],
        'OMN': [21.5126, 55.9233],
        'KWT': [29.3117, 47.4818],
        'BHR': [25.9304, 50.6378],
        'IRN': [32.4279, 53.6880],
        'IRQ': [33.2232, 43.6793],
        'PAK': [30.3753, 69.3451],
        'AFG': [33.9391, 67.7100],
        'BDI': [-3.3731, 29.9189],
        'UGA': [1.3733, 32.2903],
        'TZA': [-6.3690, 34.8888],
        'GHA': [7.9465, -1.0232],
        'CIV': [7.5399, -5.5471],
        'SEN': [14.4974, -14.4524],
        'DZA': [28.0339, 1.6596],
        'NPL': [28.3949, 84.1240],
        'LKA': [7.8731, 80.7718],
        'MMR': [21.9162, 95.9560],
        'KAZ': [48.0196, 66.9237],
        'UZB': [41.3775, 64.5853],
        'TJK': [38.8610, 71.2761],
        'KGZ': [41.2044, 74.7661],
        'MNG': [46.8625, 103.8467],
        'ISR': [31.0461, 34.8516],
        'JOR': [30.5852, 36.2384],
        'LBN': [33.8547, 35.8623],
        'SYR': [34.8021, 38.9968],
        'YEM': [15.5527, 48.5164],
        'MKD': [41.6086, 21.7453],
        'TWN': [23.6978, 120.9605],
        'CYM': [19.5133, -80.5660],
        'BHS': [25.0343, -77.3963],
        'JAM': [18.1096, -77.2975],
        'DOM': [18.7357, -70.1627],
        'HTI': [18.9712, -72.2852],
        'CUB': [21.5218, -77.7812],
        'GTM': [15.7835, -90.2308],
        'SLV': [13.7942, -88.8965],
        'HND': [15.2000, -86.2419],
        'NIC': [12.8654, -85.2072],
        'CRI': [9.7489, -83.7534],
        'PAN': [8.5380, -80.7821],
        'LUX': [49.8153, 6.1296],
        'MLT': [35.9375, 14.3754],
        'AND': [42.5063, 1.5218],
        'MCO': [43.7384, 7.4246],
        'SMR': [43.9333, 12.4500],
        'VAT': [41.9029, 12.4534],
        'MDA': [47.4116, 28.3699],
    }
    return coords

try:
    # --- Load Data ---
    df = load_data()
    coords_dict = get_country_coords()
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