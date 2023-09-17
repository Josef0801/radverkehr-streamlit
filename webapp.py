import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import numpy as np
import boto3
from io import BytesIO
import pydeck as pdk

# Initialize session state if it hasn't been initialized yet
if 'dataframes' not in st.session_state:
    # Your S3 Bucket details
    bucket_name = "nextbikeapibucket"
    file_key = "joined_left_station.pkl"

    # Your AWS Credentials
    aws_access_key_id = st.secrets["aws_access_key_id"]
    aws_secret_access_key = st.secrets["aws_secret_access_key"]
    aws_session_token = None  # Optional, use only if you are using temporary credentials

    # Initialize a session using Amazon S3
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )

    # Create S3 client
    s3 = session.client('s3')

    with st.spinner('Loading NextBikeAPI data'):
        # Use S3 client to fetch the object
        s3_object = s3.get_object(Bucket=bucket_name, Key=file_key)

        # Read the Body of the S3 object (which is a stream)
        s3_object_content = s3_object['Body'].read()

        # Deserialize the object content to a Python dictionary
        dataframes = pickle.loads(BytesIO(s3_object_content).read())

        # Save the dataframes into session_state
        st.session_state.dataframes = dataframes
        
if 'stations_lat' not in st.session_state:
    with open('stations_lat.pkl', 'rb') as f:
        st.session_state.stations_lat = pickle.load(f)
 
if 'stations_long' not in st.session_state:
    with open('stations_long.pkl', 'rb') as f:
        st.session_state.stations_long = pickle.load(f)

if 'routes_df' not in st.session_state:
    routes_df = pd.read_csv("routes_dataframe.csv")
    routes_df["date"] = pd.to_datetime(routes_df["date"])
    st.session_state.routes_df = routes_df

dataframes = st.session_state.dataframes

print("Loaded successfulle")
# Convert 'date_time' column to datetime type in all DataFrames
for df in dataframes.values():
    df['date_time'] = pd.to_datetime(df['date_time'])

# Multiselect widget to select the DataFrames
selected_dfs = st.sidebar.multiselect('Select the DataFrames', options=list(dataframes.keys()))

# Date input widgets to select the time period
start_date = st.sidebar.date_input('Start date', value=pd.to_datetime('2023-06-20').date())
end_date = st.sidebar.date_input('End date', value=pd.to_datetime('2023-09-07').date())
num_days = (end_date - start_date).days


popularity_data = []
popularity_data_hour = pd.DataFrame(columns=["lat", "long", "popularity", "hour"])
for key in dataframes:
    df = dataframes[key]

    # Filter the DataFrame based on the selected time period
    df = df[(df['date_time'].dt.date >= start_date) & (df['date_time'].dt.date <= end_date)]
    popularity = sum(df["bikes_joined"])
    popularity += (-1 * sum(df.bikes_left))
    popularity_data.append((st.session_state.stations_lat[key], st.session_state.stations_long[key], popularity))
    
    df["day"] = df["date_time"].dt.date
    df["hour"] = df["date_time"].dt.hour

    summed_df = df.groupby("hour").sum().reset_index()

    summed_df["joined_average"] = summed_df["bikes_joined"] /81
    summed_df["left_average"] = summed_df["bikes_left"] /81
    summed_df["popularity"] = summed_df["joined_average"] - 1 * summed_df["left_average"] 
    summed_df["lat"] = st.session_state.stations_lat[key]
    summed_df["long"] = st.session_state.stations_long[key]
    
    popularity_data_hour = pd.concat([popularity_data_hour, summed_df[["lat", "long", "popularity", "hour"]]])
    
    
# Convert to DataFrame for easier processing

popularity_df = pd.DataFrame(popularity_data, columns=["lat", "lon", "popularity"])


# Normalize popularity for visualization
max_popularity = popularity_df['popularity'].max()
popularity_df['norm_popularity'] = popularity_df['popularity'] / max_popularity


# ------------------------------------ Statistics ------------------------------
st.divider()
st.title("Statistics")
st.write("""
         Here some descriptive statistics.
         """)



# Filter the data
df = st.session_state.routes_df.copy()
filtered_data = df[(df['date'].dt.date  >= start_date) & (df['date'].dt.date <= end_date)]
date_diff = (end_date - start_date).days + 1

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total number of commutes", value=len(filtered_data))
          
col2.metric("Average number of commutes per day", value=np.round(len(filtered_data)/date_diff, 2))
    


avg_duration = np.round(np.mean(filtered_data['duration']), 2)
col3.metric("Average duration", value=avg_duration)

median_duration = np.median(np.round(filtered_data['duration'], 2))
col4.metric("Median duration", value=median_duration)

st.subheader("Number of commutes")
commutes_per_day = filtered_data.groupby(filtered_data['date'].dt.date).size()
st.line_chart(commutes_per_day)

# ------------- Hour Plot

import numpy as np

st.session_state.routes_df["hour"] = st.session_state.routes_df["date"].dt.hour

st.subheader("Average number of commutes per hour")

grouped_df = np.round(st.session_state.routes_df.groupby("hour").count()/81, 2)
grouped_df["Avg. number of commutes"] = grouped_df["from"]



st.line_chart(grouped_df["Avg. number of commutes"] )

# -------------- Bar chart


# # Filter out rows with non-official 'from' and 'to' locations
# viable_routes = filtered_data[filtered_data['from'].isin(st.session_state.stations_lat ) & filtered_data['to'].isin(st.session_state.stations_lat )]

# # Group by 'from' and 'to', then calculate average duration
# grouped_df = viable_routes.groupby(['from', 'to']).agg({'duration': 'mean'}).reset_index()
# grouped_df["duration"] = grouped_df["duration"].round(2) 

# # Sort by duration and take top 10
# top_routes = grouped_df.sort_values(by='duration', ascending=False).head(100)

# # Streamlit app
# st.subheader("10 longes Routes by Average Duration")
# top_routes['label'] = top_routes['from'] + " → " + top_routes['to']

# chart = (
#     alt.Chart(top_routes)
#     .mark_bar()
#     .encode(
#         y=alt.Y('label:O', title="Route", sort='-x'), # 'O' indicates ordinal data type
#         x=alt.X('duration:Q', title="Average Duration"), # 'Q' indicates quantitative data type
#         color=alt.Color('duration:Q', scale=alt.Scale(scheme='blueorange')), # Color by duration
#         tooltip=['from', 'to', 'duration'] # Tooltip information on hover
#     )
#     .properties(width=600, height=400, title="Top 10 Routes by Average Duration")
# )

# st.altair_chart(chart, use_container_width=True)
# # Visualize the data using a bar plot
# # st.bar_chart(top_routes.set_index(['from', 'to'])['duration'])



# ------------------------------------ Circle plot

# Set Streamlit title and instructions
st.title("Popularity of Stations")
st.write("""
         The bigger and more intensive the circle, 
         the more popular the station is.
         """)

# Pydeck chart
view_state = pdk.ViewState(
    latitude=popularity_df['lat'].mean(),
    longitude=popularity_df['lon'].mean(),
    zoom=12,
    pitch=0
)

scaling_factor = 0.015 * (80 / num_days)
radius_string = f"popularity * {scaling_factor}"

layer = pdk.Layer(
    "ScatterplotLayer",
    data=popularity_df,
    get_position='[lon, lat]',
    get_radius=radius_string,
    get_fill_color="[255, (1-norm_popularity)*255, (1-norm_popularity)*255, 150]",
    pickable=True,
    auto_highlight=True
)

chart = pdk.Deck(
    initial_view_state=view_state,
    layers=[layer],
    tooltip={"text": "Popularity: {popularity}"}
)

st.pydeck_chart(chart)









# -------------------------------- ROUTE PLOT
import folium
from streamlit_folium import folium_static

st.title("Visualizing Popular Routes")
st.write("Below you can see all the routes with more than 500 commutes. The red-ness and thickness indicate the popularity")

df = st.session_state.routes_df

df = df.groupby(['from_coordinates', 'to_coordinates', 'from', 'to']).size().reset_index(name='popularity')
df = df[df["popularity"] >= 400]

df['from_coordinates'] = df['from_coordinates'].apply(eval)
df['to_coordinates'] = df['to_coordinates'].apply(eval)

df["start_coordinates"] = df["from_coordinates"].apply(lambda x: (x[1], x[0]))
df["target_coordinates"] = df["to_coordinates"].apply(lambda x: (x[1], x[0]))

# df["color"] = df["color"].apply(hex_to_rgb)
GREEN_RGB = [0, 255, 0, 40]
RED_RGB = [240, 100, 0, 40]

view_state = pdk.ViewState(latitude=50.848502, longitude=8.764375, zoom=12,
    pitch=50)

def calculate_color(value, max_value=1400):

    value = max(0, min(max_value, value))
    
    red_intensity = int(255 * (value / max_value))
    
    return (red_intensity, 255 - red_intensity, 255 - red_intensity)

df["color"] = df["popularity"].apply(lambda x: calculate_color(x))

arc_layer = pdk.Layer(
    "ArcLayer",
    data=df,
    get_width="popularity * 0.003",
    get_source_position="start_coordinates",
    get_target_position="target_coordinates",
    get_tilt=15,
    get_source_color="color",
    get_target_color="color",
    pickable=True,
    auto_highlight=True,
)


chart = pdk.Deck(
    initial_view_state=view_state,
    layers=arc_layer,
tooltip={"text": "From:\t{from}\nTo:\t{to}\nPopularity:\t{popularity}"}
)

st.pydeck_chart(chart)


# ------------------------------------  Arrival and Departures ------------------------------

st.divider()
st.title("Arrival and Departures")
st.write("""
         The following plots show the departures and arrivals and the selected stations.
         """)
    
# Iterate over the selected DataFrames
for key in selected_dfs:
    df = dataframes[key]

    # Filter the DataFrame based on the selected time period
    df = df[(df['date_time'].dt.date >= start_date) & (df['date_time'].dt.date <= end_date)]

    # Set the 'date_time' column as the index
    df.set_index('date_time', inplace=True)

    # Create a new figure
    plt.figure()
    plt.rcParams['figure.figsize'] = (20, 10)

    # Plot the 'capacity added' and 'capacity lost' values over time
    df.plot(y=['bikes_joined', 'bikes_left'], marker='o')


    # Generate the list of timestamps at noon and midnight for each day
    noon_and_midnight = pd.date_range(df.index.min().date(), df.index.max().date() + pd.Timedelta(days=1), freq='12H')

#     # Add vertical lines at each timestamp in the list
#     for timestamp in noon_and_midnight:
#         if timestamp.hour == 0:  # 12 AM
#             plt.axvline(timestamp, color='darkgrey', linestyle='--', linewidth=2)
#         else:  # 12 PM
#             plt.axvline(timestamp, color='black', linestyle='--', linewidth=2)

    # Set the labels and title
    plt.ylim(-10, 10)
    plt.xlabel('Datum & Uhrzeit')
    plt.ylabel('Anzahl der hinzugefügten und abgenommenen Bikes')
    plt.title('Zuwachs und Abnahme {}'.format(key))

    # Add custom legends for the vertical lines and the plot
    plt.legend(['bikes_joined', 'bikes_left', '12 AM', '12 PM'])

    # Ensure the x-axis dates are properly formatted
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    # Show the plot
    st.pyplot(plt)
