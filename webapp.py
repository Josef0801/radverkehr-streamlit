import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle

import boto3
from io import BytesIO

# Initialize session state if it hasn't been initialized yet
if 'dataframes' not in st.session_state:
    # Your S3 Bucket details
    bucket_name = "nextbikeapibucket"
    file_key = "saved_dictionary.pkl"

    # Your AWS Credentials
    aws_access_key_id = st.secrets["aws_access_key_id"]
    aws_secret_access_key = st.secrets["aws_secret_access_key"]
    # aws_access_key_id = "AKIAWDTQFMVEMYWGYRFJ"
    # aws_secret_access_key = "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"
    # aws_session_token = None  # Optional, use only if you are using temporary credentials

    # Initialize a session using Amazon S3
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )

    # Create S3 client
    s3 = session.client('s3')

    # Use S3 client to fetch the object
    s3_object = s3.get_object(Bucket=bucket_name, Key=file_key)

    # Read the Body of the S3 object (which is a stream)
    s3_object_content = s3_object['Body'].read()

    # Deserialize the object content to a Python dictionary
    dataframes = pickle.loads(BytesIO(s3_object_content).read())
    
    # Save the dataframes into session_state
    st.session_state.dataframes = dataframes

# Retrieve dataframes from session_state
dataframes = st.session_state.dataframes

# dataframes = pickle.loads(BytesIO(s3_object_content).read())

# with open('saved_dictionary.pkl', 'rb') as f:
#     dataframes = pickle.load(f)

# Convert 'date_time' column to datetime type in all DataFrames
for df in dataframes.values():
    df['date_time'] = pd.to_datetime(df['date_time'])

# Multiselect widget to select the DataFrames
selected_dfs = st.sidebar.multiselect('Select the DataFrames', options=list(dataframes.keys()))

# Date input widgets to select the time period
start_date = st.sidebar.date_input('Start date', value=pd.to_datetime('2023-06-20').date())
end_date = st.sidebar.date_input('End date', value=pd.to_datetime('2023-08-01').date())

# Iterate over the selected DataFrames
for key in selected_dfs:
    df = dataframes[key]
    print("COLUMNS :"+str(df.columns))
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

    
import streamlit as st
import pandas as pd
import pydeck as pdk

data = pd.read_csv("route_data.csv", index_col=0)

# Create a pydeck Layer
layer = pdk.Layer(
    "PathLayer",
    data,
    get_path=["start_coordinates", "end_coordinates"],
    get_width=5,
    get_color=[255, 0, 0],
    pickable=True,
    auto_highlight=True
)

# Set the initial viewport for the map
view_state = pdk.ViewState(
    latitude=data.start_lat.mean(),
    longitude=data.start_lon.mean(),
    zoom=5,
    min_zoom=15,
    max_zoom=20,
    pitch=40.5,
    bearing=-27.36,
)

# Create a pydeck Deck and display it in Streamlit
deck = pdk.Deck(layers=[layer], initial_view_state=view_state)
st.pydeck_chart(deck)
