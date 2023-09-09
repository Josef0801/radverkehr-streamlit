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
    st.session_state.routes_df = pd.read_csv("routes_dataframe.csv")


dataframes = st.session_state.dataframes = dataframes


# Convert 'date_time' column to datetime type in all DataFrames
for df in dataframes.values():
    df['date_time'] = pd.to_datetime(df['date_time'])

# Multiselect widget to select the DataFrames
selected_dfs = st.sidebar.multiselect('Select the DataFrames', options=list(dataframes.keys()))

# Date input widgets to select the time period
start_date = st.sidebar.date_input('Start date', value=pd.to_datetime('2023-06-20').date())
end_date = st.sidebar.date_input('End date', value=pd.to_datetime('2023-09-07').date())
num_days = (end_date - start_date).days
print("NUM DAYS")
print(num_days)

popularity_data = []
for key in dataframes:
    df = dataframes[key]

    # Filter the DataFrame based on the selected time period
    df = df[(df['date_time'].dt.date >= start_date) & (df['date_time'].dt.date <= end_date)]
    popularity = sum(df["bikes_joined"])
    popularity += (-1 * sum(df.bikes_left))
    popularity_data.append((st.session_state.stations_lat[key], st.session_state.stations_long[key], popularity))
       

# Convert to DataFrame for easier processing
popularity_df = pd.DataFrame(popularity_data, columns=["lat", "lon", "popularity"])

# Normalize popularity for visualization
max_popularity = popularity_df['popularity'].max()
popularity_df['norm_popularity'] = popularity_df['popularity'] / max_popularity

print("POP DF")
print(popularity_df)
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

st.divider()
st.title("Popularity of routes")
st.write("""
         The bigger and more intensive the circle, 
         the more popular the location is.
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

    # Add vertical lines at each timestamp in the list
    for timestamp in noon_and_midnight:
        if timestamp.hour == 0:  # 12 AM
            plt.axvline(timestamp, color='darkgrey', linestyle='--', linewidth=2)
        else:  # 12 PM
            plt.axvline(timestamp, color='black', linestyle='--', linewidth=2)

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


    
import pandas as pd
import streamlit as st
import pydeck as pdk

# Sample Data
data = pd.DataFrame({
    'from': ['BIKE 385534', 'Friedrichplatz', 'Südbahnhof Westseite', 'BIKE 385822', 'Ketzerbach/Zwischenhausen'],
    'to': ['Universitätsstraße/Bibliothek Jura', 'Biegenstraße/Cineplex', 'Wilhelmsplatz', 'Frankfurter Straße/Theater', 'Hauptbahnhof'],
    'date': ['2023-06-19 22:27:31.019026', '2023-06-19 22:35:49.360453', '2023-06-19 22:39:49.184764', '2023-06-19 22:43:49.184131', '2023-06-19 22:43:49.184131'],
    'from_coordinates': ['(50.807716, 8.769865)', '(50.80327, 8.76406)', '(50.79527, 8.762049)', '(50.8034, 8.767411)', '(50.81395, 8.76616)'],
    'to_coordinates': ['(50.807063, 8.769918)', '(50.808867, 8.77305)', '(50.804725, 8.759248)', '(50.798841, 8.762144)', '(50.819957, 8.773736)']
})

# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Filter by the date
start_date = '2023-06-19'  # Update as per your need
end_date = '2023-06-20'    # Update as per your need
filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

# Group by the from_coordinates and to_coordinates and get the popularity score
grouped_data = filtered_data.groupby(['from_coordinates', 'to_coordinates']).size().reset_index(name='popularity')

# Convert string representations to actual tuples
grouped_data['from_coordinates'] = grouped_data['from_coordinates'].apply(eval)
grouped_data['to_coordinates'] = grouped_data['to_coordinates'].apply(eval)

# Create a 'path' column
grouped_data['path'] = grouped_data.apply(lambda row: [row['from_coordinates'], row['to_coordinates']], axis=1)

# Set up pydeck chart
view_state = pdk.ViewState(latitude=50.807894, longitude=8.766971, zoom=12, pitch=0)

# Create PathLayer
routes_layer = pdk.Layer(
    type="PathLayer",
    data=grouped_data,
    get_path="path",
    get_width="popularity * 10000",
    get_color="[255, 100, 100, 200]",  # Adjust as per your needs
    pickable=True,
    width_scale=20,
    width_min_pixels=20,
)
# This will return a Series of boolean values. True if the value in 'path' is a list of lists, False otherwise.
is_list_of_lists = grouped_data['path'].apply(lambda x: isinstance(x, list) and all(isinstance(i, list) for i in x))

# Print out any rows where 'path' is not a list of lists
incorrect_format = grouped_data[~is_list_of_lists]
st.write(incorrect_format)

chart = pdk.Deck(initial_view_state=view_state, layers=[routes_layer], tooltip={"text": "Popularity: {popularity}"})

st.pydeck_chart(chart)




# # Prepare 'start_coordinates' and 'end_coordinates' columns in the format PathLayer expects
# data['start_coordinates'] = data.apply(lambda row: [row['start_lon'], row['start_lat']], axis=1)
# data['end_coordinates'] = data.apply(lambda row: [row['end_lon'], row['end_lat']], axis=1)

# # Normalize the count column for better visual representation
# max_count = data['count'].max()
# min_count = data['count'].min()
# data['normalized_count'] = 5 + (data['count'] - min_count) * 20 / (max_count - min_count)

# # Create interpolated RGB colors
# data['color_r'] = np.interp(data['normalized_count'], [0, 1], [0, 255]).astype(int)
# data['color_g'] = np.interp(data['normalized_count'], [0, 1], [0, 0]).astype(int)
# data['color_b'] = np.interp(data['normalized_count'], [0, 1], [255, 0]).astype(int)

# # Create a pydeck Layer with the new color scheme
# layer = pdk.Layer(
#     "PathLayer",
#     data,
#     get_path='[start_coordinates, end_coordinates]',
#     get_width='normalized_count * 50 + 2',  # Adjust the line thickness here
#     get_color='[color_r, color_g, color_b, 150]',  # Using interpolated RGB values
#     width_scale=2,
#     width_min_pixels=2,
#     pickable=True,
#     auto_highlight=True
# )

# # Set the initial viewport for the map
# view_state = pdk.ViewState(
#     latitude=data['start_lat'].mean(),
#     longitude=data['start_lon'].mean(),
#     zoom=12,
#     min_zoom=10,
#     max_zoom=20,
#     pitch=20,
#     bearing=-27.36,
# )

# # Create a pydeck Deck and display it in Streamlit
# deck = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v9")
# st.pydeck_chart(deck)
