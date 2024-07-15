import pickle
import os
import requests
import json
import time
import warnings
import joblib
import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import seaborn as sns
from datetime import datetime, timedelta, date
import datetime as dt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import openai
import io
import math
from cryptography.fernet import Fernet

####### utils.py has bulky functions
from utils import *
##

warnings.filterwarnings('ignore')

config_file_path = 'data/dev_config.json'
# config_file_path = '/root/cora_trail_status_classifier/data/prod_config.json'

with open(config_file_path, 'r') as config_file:
    config = json.load(config_file)

data_directory = config['paths']['data_directory']
os.chdir(data_directory)

# lookback_days_list = [2, 3, 4, 5, 6, 7, 11, 18]

s3_client = boto3.client('s3')
s3 = boto3.client('s3')
bucket_name = 'mtb-trail-condition-predictions'

### APIS (ENCRYPTED) 
def load_api_key(encryption_key_path, encrypted_api_key_path):
    with open(encryption_key_path, "rb") as f:
        encryption_key = f.read()
    cipher_suite = Fernet(encryption_key)
    with open(encrypted_api_key_path, "rb") as f:
        encrypted_api_key = f.read()
    decrypted_api_key = cipher_suite.decrypt(encrypted_api_key)
    return decrypted_api_key.decode()

def load_all_api_keys():
    keys = {
        "openweather_api_key": load_api_key("creds/encryption_openweather_key.txt", "creds/encrypted_openweather_api_key.txt"),
        "gmail_api_key": load_api_key("creds/encryption_gmail_key.txt", "creds/encrypted_gmail_api_key.txt"),
        "openai_api_key": load_api_key("creds/encryption_openai_key.txt", "creds/encrypted_openai_api_key.txt"),
        "accuweather_api_key": load_api_key("creds/encryption_accuweather_key.txt", "creds/encrypted_accuweather_api_key.txt")
    }
    return keys

api_keys = load_all_api_keys()
openweather_api_key = api_keys["openweather_api_key"]
gmail_api_key = api_keys["gmail_api_key"]
openai_api_key = api_keys["openai_api_key"]
accuweather_api_key = api_keys["accuweather_api_key"]

file_key = 'data/trail_locations.csv'
obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
print("Reading in file from S3 bucket")
df_trail_locations = pd.read_csv(obj['Body'])
print(df_trail_locations.head(15))

exclude = "minutely,hourly,alerts"
pickle_file = 'weather_data.pickle'

##### ACCUWEATHER

# Function to get the location key using latitude and longitude
def get_location_key(api_key, latitude, longitude):
    location_url = f"http://dataservice.accuweather.com/locations/v1/cities/geoposition/search"
    params = {
        'apikey': api_key,
        'q': f"{latitude},{longitude}"
    }
    response = requests.get(location_url, params=params)
    response.raise_for_status()  # Check for errors
    location_data = response.json()
    return location_data['Key']


def accuweather_data(df_trail_locations, api_key):
    pickle_file = 'accuweather_data.pickle'
    csv_file_key = 'data/accuweather_data.csv'

    print("Fetching new hourly data for:", datetime.now().date())

    # Check if the pickle file has been modified in the past hour
    if os.path.exists(pickle_file):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(pickle_file))
        if datetime.now() - file_mod_time < timedelta(minutes=50):
            print("Pickle file has been modified in the past hour. Loading existing data.")
            with open(pickle_file, 'rb') as f:
                hourly_weather = pickle.load(f)
            return hourly_weather
        else:
            print("Pickle file is older than 50 minutes. Fetching new data.")
            hourly_weather = pd.DataFrame()
    else:
        hourly_weather = pd.DataFrame()

    # Loop through each trail location
    all_data = []
    for index, row in df_trail_locations.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        trail = row['Trail']
        location_key = row['Accuweather Location Key']
        print(trail, ": Location Key: ", location_key)
        time.sleep(1)
        
        # Fetch hourly data
        hourly_url = f"http://dataservice.accuweather.com/currentconditions/v1/{location_key}/historical"
        params = {
                'apikey': api_key,
                'language': 'en-us',  # Specify the language
                'details': 'true'     # Include full details in the response
            }
            
        response = requests.get(hourly_url, params=params)
        # print(response)
        response.raise_for_status()
        new_data = response.json()

        # Process the data into a DataFrame
        run_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for hour_data in new_data:
            observation_time = datetime.fromisoformat(hour_data['LocalObservationDateTime'][:-6])
            formatted_time = observation_time.strftime('%Y-%m-%d %H:00')
            data = {
                'HOUR': formatted_time,
                'Trail': trail,
                'MAX TEMPERATURE': hour_data['Temperature']['Imperial']['Value'],
                'TOTAL PRECIPITATION': hour_data.get('PrecipitationSummary', {}).get('Precipitation', {}).get('Imperial', {}).get('Value', 0),
                'DEW_POINT': hour_data['DewPoint']['Imperial']['Value'],
                'run_datetime': run_datetime
            }
            all_data.append(data)
            print(data)

    new_hourly_weather = pd.DataFrame(all_data)

    # Load existing CSV data from S3 if it exists
    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=csv_file_key)
        existing_hourly_weather = pd.read_csv(obj['Body'])
        hourly_weather = pd.concat([existing_hourly_weather, new_hourly_weather])
        print("added to Accuweather S3 CSV")
    except s3_client.exceptions.NoSuchKey:
        hourly_weather = new_hourly_weather
        print("NO SUCH KEY")

    # Remove duplicates, keeping the most recent 'run_datetime'
    hourly_weather = hourly_weather.sort_values(by='run_datetime').drop_duplicates(subset=['HOUR', 'Trail'], keep='last')

    # Filter data to the most recent 10 days
    ten_days_ago = datetime.now() - timedelta(days=10)
    hourly_weather['HOUR'] = pd.to_datetime(hourly_weather['HOUR'])
    hourly_weather = hourly_weather[hourly_weather['HOUR'] >= ten_days_ago]

    # Save the updated data back to S3
    hourly_weather.to_csv(csv_file_key, index=False)
    s3_client.upload_file(csv_file_key, bucket_name, csv_file_key)

    # Save to pickle file
    with open(pickle_file, 'wb') as f:
        pickle.dump(hourly_weather, f)
    print("Hourly data updated and saved to pickle file and S3.")

    return hourly_weather

hourly_accuweather = accuweather_data(df_trail_locations, accuweather_api_key)

#######

hourly_weather = update_hourly_weather_data(df_trail_locations, openweather_api_key)

print("###########################")
print("Accuweather")
print(hourly_accuweather.head(5))
print("LEN ACCUWEATHER", len(hourly_accuweather))

print("###########################")
print("Open Weather")
print(hourly_weather.head(5))

########
# END
#########


# pickle_file = 'historical_one_week_all_trails.pickle'


# historical_one_week_all_trails = fetch_historical_weather_data(pickle_file, df_trail_locations, openweather_api_key)

# historical_one_week_all_trails.to_csv("historical_one_week_all_trails.csv")


# yesterday = datetime.now() - timedelta(days=1)
# yesterday_date_str = yesterday.strftime('%Y-%m-%d')
# historical_one_week_all_trails['DATE'] = pd.to_datetime(historical_one_week_all_trails['DATE'], format='%Y-%m-%d')
# historical_one_week_all_trails_filtered = historical_one_week_all_trails[historical_one_week_all_trails['DATE'] < yesterday]

# COMBINE WEATHER DFs
# weather_append = pd.concat([hourly_weather, historical_one_week_all_trails_filtered])
hourly_accuweather['DATE'] = hourly_accuweather['HOUR'].dt.strftime('%Y-%m-%d')
hourly_accuweather['HOUR'] = hourly_accuweather['HOUR'].dt.strftime('%H')
hourly_accuweather.to_csv("hourly_accuweather.csv")


weather_append = hourly_accuweather.drop_duplicates(subset=['DATE', 'Trail', 'HOUR', 'TOTAL PRECIPITATION'], keep='first')
weather_append['DATE'] = pd.to_datetime(weather_append['DATE'])
weather_append['DATE'] = weather_append['DATE'].dt.date
print("PRINTING NEW WEATHER DATA #######")
print(weather_append.sort_values(by='DATE', ascending=True))
print("---------------")
weather_sorted = weather_append.sort_values(by='DATE', ascending=False)
weather_sorted.set_index('DATE', inplace=True)
print("---------------")

column_names = {'Trail': 'trail', 'DATE': 'DATE', 'MAX TEMPERATURE': 'TMAX', 'TOTAL PRECIPITATION': 'PRCP', 'DEW_POINT': 'DEW_POINT', 'PRECIPITATION PROBABILITY': 'PROB_ADJ'} # 'SNOW_FLAG': 'SNOW_FLAG'}
weather_sorted = weather_sorted.reset_index().rename(columns=column_names)
print(weather_sorted.head(5))
weather_data_main = weather_sorted.copy()
print("---------------")

weather_data_main = weather_data_main.sort_values(['trail','DATE'])
weather_data_main.to_csv("weather_data_main.csv")
### ROLLING METRICS CALCULATED

weather_data_hourly = weather_data_main[weather_data_main["HOUR"].notnull()]
# weather_data_daily = weather_data_main[weather_data_main["HOUR"].isnull()]

yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)

print("Hourly Data:\n", weather_data_hourly.sort_values(["trail", "DATE"], ascending = [False, False]).head(35))
lookback_hours_list = [4, 8, 16, 24, 48, 72, 96, 120, 144, 168]

# Assume weather_data_hourly and weather_data_daily are already defined
# daily_df = calculate_rolling_metrics_daily(weather_data_daily, lookback_days_list)
hourly_df = calculate_rolling_metrics_hourly(weather_data_hourly, lookback_hours_list)

# daily_df_trim = daily_df.groupby('trail', as_index=False).apply(lambda x: x.loc[x['DATE'].idxmax()]).reset_index(drop=True)
# daily_df_trim = daily_df.sort_values(by='DATE', ascending=False).drop_duplicates(subset = 'trail').reset_index(drop=True)
hourly_df_trim = hourly_df.sort_values(by=['DATE', 'HOUR'], ascending=[False, False]).drop_duplicates(subset = 'trail').reset_index(drop=True)
# final_df = pd.merge(daily_df_trim, hourly_df_trim, on='trail')
final_df = hourly_df_trim.copy()
print("FINAL DF")
print(final_df.head(10))
final_df.to_csv("final_df.csv")

### 

df_class = final_df[['trail', 
                     'PRCP_24h', 'PRCP_48h', 'PRCP_72h', 'PRCP_120h', 'PRCP_168h', 
                     'PRCP_4h', 'PRCP_8h', 'PRCP_16h',
                     'TMAX_24h', 'TMAX_48h', 'TMAX_72h', 'TMAX_120h', 'TMAX_168h', 
                     'TMAX_4h', 'TMAX_8h', 'TMAX_16h',
                     'DEW_POINT_24h', 'DEW_POINT_48h', 'DEW_POINT_72h', 'DEW_POINT_120h', 'DEW_POINT_168h', 
                     'DEW_POINT_4h', 'DEW_POINT_8h', 'DEW_POINT_16h'
                     ]]


print("Dataframe for Classification")
print(df_class.head(15))

#####
#START CLASSIFICATION
#####

sheet_id = '1IrZgmVjHmFkdxxM_65XzKW_nt8p8LCIHgkqtbjY4QJE'
sheet_name = 'trail_adjustments'
url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
df_gsheet_trail_adjustments = pd.read_csv(url)
trail_adjustments = dict(zip(df_gsheet_trail_adjustments['Trail'], df_gsheet_trail_adjustments['PRCP_ADJ_VALUE']))

sheet_id = '1IrZgmVjHmFkdxxM_65XzKW_nt8p8LCIHgkqtbjY4QJE'
sheet_name = 'if_statements'
url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
df_gsheet_if_statements = pd.read_csv(url)
print(df_gsheet_if_statements.head(10))

def adjust_prcp(prcp, tmax, dew_point, trail):
    # Adjust based on TMAX
    if tmax > 90:
        prcp *= 0.90
    elif tmax > 85:
        prcp *= 0.95   
    elif tmax > 70:
        prcp *= 1.0 
    elif tmax <= 45:
        prcp *= 1.3
    elif tmax <= 55:
        prcp *= 1.2    
    elif tmax <= 65:
        prcp *= 1.1 

    dew_point_temp_diff = tmax - dew_point

    # Adjust based on DEW_POINT DIFF
    if dew_point_temp_diff < 5:
        prcp *= 1.3
    elif dew_point_temp_diff < 10:
        prcp *= 1.2
    elif dew_point_temp_diff < 15:
        prcp *= 1.1  
    elif dew_point_temp_diff < 20:
        prcp *= 0.95
    elif dew_point_temp_diff < 25:
        prcp *= 0.9 
    
    # Apply trail adjustment if the trail is in the adjustment list
    if trail in trail_adjustments:
        prcp *= trail_adjustments[trail]

    return prcp


status_scores = {
    'DEFINITE CLOSE': 5,
    'LIKELY CLOSE': 4,
    'LIKELY WET/OPEN': 3,
    'LIKELY OPEN': 2,
    'DEFINITE OPEN': 1
}

def trail_status(row):
    prcp_4h = row['PRCP_4h']
    prcp_8h = row['PRCP_8h']
    prcp_16h = row['PRCP_16h']
    prcp_24h = row['PRCP_24h']
    prcp_48h = row['PRCP_48h']
    prcp_72h = row['PRCP_72h']
    prcp_120h = row['PRCP_120h']
    prcp_168h = row['PRCP_168h']
    tmax_4h = row['TMAX_4h']
    tmax_8h = row['TMAX_8h']
    tmax_16h = row['TMAX_16h']
    tmax_24h = row['TMAX_24h']
    tmax_48h = row['TMAX_48h']
    tmax_72h = row['TMAX_72h']
    tmax_120h = row['TMAX_120h']
    tmax_168h = row['TMAX_168h']
    dew_point_4h = row['DEW_POINT_4h']
    dew_point_8h = row['DEW_POINT_8h']
    dew_point_16h = row['DEW_POINT_16h']
    dew_point_24h = row['DEW_POINT_24h']
    dew_point_48h = row['DEW_POINT_48h']
    dew_point_72h = row['DEW_POINT_72h']
    dew_point_120h = row['DEW_POINT_120h']
    dew_point_168h = row['DEW_POINT_168h']
    trail = row['trail']

    # Adjust precipitation values
    prcp_4h = adjust_prcp(prcp_4h, tmax_4h, dew_point_4h, trail)
    prcp_8h = adjust_prcp(prcp_8h, tmax_8h, dew_point_8h, trail)
    prcp_16h = adjust_prcp(prcp_16h, tmax_16h, dew_point_16h, trail)
    prcp_24h = adjust_prcp(prcp_24h, tmax_24h, dew_point_24h, trail)
    prcp_48h = adjust_prcp(prcp_48h, tmax_48h, dew_point_48h, trail)
    prcp_72h = adjust_prcp(prcp_72h, tmax_72h, dew_point_72h, trail)
    prcp_120h = adjust_prcp(prcp_120h, tmax_120h, dew_point_120h, trail)
    prcp_168h = adjust_prcp(prcp_168h, tmax_168h, dew_point_168h, trail)

    dimensions = ['prcp_4h', 'prcp_8h', 'prcp_16h', 'prcp_24h', 'prcp_48h', 'prcp_72h', 'prcp_120h', 'prcp_168h']
    status_levels = ['DEFINITE CLOSE', 'LIKELY CLOSE', 'LIKELY WET/OPEN', 'LIKELY OPEN', 'DEFINITE OPEN']

    # Initialize dictionaries to store the threshold values
    greater_than = {dim: {} for dim in dimensions}
    less_than = {dim: {} for dim in dimensions}

    # Populate the dictionaries with the threshold values
    for dim in dimensions:
        for status in status_levels:
            greater_than[dim][status] = df_gsheet_if_statements.query(f"`Metric Direction` == 'Greater than' and `PRCP Dimension` == '{dim}'")[status].values[0]
            less_than[dim][status] = df_gsheet_if_statements.query(f"`Metric Direction` == 'Less than' and `PRCP Dimension` == '{dim}'")[status].values[0]

    def get_trail_status(prcp_values):
        status_counts = []

        for status in status_levels:
            count = 0

            try:
                if greater_than['prcp_4h'][status] <= prcp_values['prcp_4h'] <= less_than['prcp_4h'][status]:
                    count += 0.25
                if greater_than['prcp_8h'][status] <= prcp_values['prcp_8h'] <= less_than['prcp_8h'][status]:
                    count += 0.5
                if greater_than['prcp_16h'][status] <= prcp_values['prcp_16h'] <= less_than['prcp_16h'][status]:
                    count += 0.25
                if greater_than['prcp_24h'][status] <= prcp_values['prcp_24h'] <= less_than['prcp_24h'][status]:
                    count += 1
                if greater_than['prcp_48h'][status] <= prcp_values['prcp_48h'] <= less_than['prcp_48h'][status]:
                    count += 1.5
                if greater_than['prcp_72h'][status] <= prcp_values['prcp_72h'] <= less_than['prcp_72h'][status]:
                    count += 1
                if greater_than['prcp_120h'][status] <= prcp_values['prcp_120h'] <= less_than['prcp_120h'][status]:
                    count += 1
                if greater_than['prcp_168h'][status] <= prcp_values['prcp_168h'] <= less_than['prcp_168h'][status]:
                    count += 1
            except KeyError as e:
                print(f"KeyError: {e}. prcp_values: {prcp_values}")

            status_counts.append({'status': status, 'count': count})

        # Convert the list to a DataFrame
        df_status_counts = pd.DataFrame(status_counts)

        # Add a count of X to 'LIKELY WET/OPEN' before calculation
        # Doing this to make the algorithim more conservative and less prone towards creating DEFINITE Classifications
        # In other words, make the algorithim have to "fight" a bit more to make definite conclusions.
        # This will artificially bring down the weighted average scores, and make LIKELY WET/OPEN Classifications more common
        df_status_counts.loc[df_status_counts['status'] == 'LIKELY WET/OPEN', 'count'] += 1.0
        
        total_weighted_score = 0
        total_count = 0

        for _, row in df_status_counts.iterrows():
            status = row['status']
            count = row['count']
            total_weighted_score += count * status_scores[status]
            total_count += count

        weighted_average_score = total_weighted_score / total_count if total_count != 0 else 0

        # Find the closest and second closest status based on the weighted average score
        sorted_statuses = sorted(status_scores.keys(), key=lambda k: abs(status_scores[k] - weighted_average_score))
        closest_status = sorted_statuses[0]
        next_closest_status = sorted_statuses[1]

        return closest_status, next_closest_status
    
    prcp_values = {
        'prcp_4h': prcp_4h,
        'prcp_8h': prcp_8h,
        'prcp_16h': prcp_16h,
        'prcp_24h': prcp_24h,
        'prcp_48h': prcp_48h,
        'prcp_72h': prcp_72h,
        'prcp_120h': prcp_120h,
        'prcp_168h': prcp_168h
    }
    print("processing this row of data", row[['trail', 'PRCP_4h', 'PRCP_8h', 'PRCP_16h', 'PRCP_24h', 'PRCP_48h', 'PRCP_72h', 'PRCP_120h', 'PRCP_168h']])
    print("Result: ", get_trail_status(prcp_values))
    
    closest_status, next_closest_status = get_trail_status(prcp_values)
    return pd.Series({'trail_status': closest_status, 'next_closest_trail_status': next_closest_status})

# Applying the updated function to the DataFrame
final_df[['trail_status', 'next_closest_trail_status']] = final_df.apply(trail_status, axis=1)
print(final_df[['trail', 'PRCP_24h', 'PRCP_8h', 'PRCP_48h', 'PRCP_16h', 'PRCP_72h', 'PRCP_120h', 'PRCP_168h', 'TMAX_24h', 'DEW_POINT_24h', 'trail_status', 'next_closest_trail_status']])


# Initialize the S3 client
s3_client = boto3.client('s3')
bucket_name = 'mtb-trail-condition-predictions'
log_file_key = 'trail_conditions_log.csv'


def get_current_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:00:00')

def load_existing_log():
    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=log_file_key)
        log_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        return log_df
    except s3_client.exceptions.NoSuchKey:
        return pd.DataFrame(columns=['trail', 'PRCP_4h', 'PRCP_8h', 'PRCP_16h', 'PRCP_24h', 'PRCP_48h', 'PRCP_72h', 'PRCP_120h', 'PRCP_168h', 'trail_status', 'next_closest_trail_status', 'timestamp'])

def save_log_to_s3(log_df):
    csv_buffer = io.StringIO()
    log_df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket_name, Key=log_file_key, Body=csv_buffer.getvalue())

def append_to_log(final_df):
    current_timestamp = get_current_timestamp()
    final_df['timestamp'] = current_timestamp
    
    log_df = load_existing_log()
    
    # Check if the log is empty or if the current timestamp and trail combination doesn't exist in the log
    if log_df.empty or not ((log_df['timestamp'] == current_timestamp) & (log_df['trail'] == final_df['trail'].values[0])).any():
        log_df = pd.concat([log_df, final_df], ignore_index=True)
        save_log_to_s3(log_df)
    else:
        print("Log for the current timestamp already exists.")

    return log_df

# Append the final_df to log
log_df = append_to_log(final_df[['trail', 'PRCP_4h', 'PRCP_8h', 'PRCP_16h', 'PRCP_24h', 'PRCP_48h', 'PRCP_72h', 'PRCP_120h', 'PRCP_168h', 'trail_status', 'next_closest_trail_status',]]) #timestamp is created, not inputted

# To send full timestamp for QA purposes to Nathan at CORA
log_df['timestamp_with_date'] = log_df['timestamp']

print("LOG DF", log_df.head())
#### VIEW LOG

#### VIEW LOG
print("VIEW LOG ######")
# Get current timestamp and past 24 hours timestamp
current_timestamp = datetime.now().replace(minute=59, second=0, microsecond=0)
past_24_hours = current_timestamp - timedelta(hours=24)

# Filter log_df based on timestamp in past 24 hours and sort it
# the script runs once per hour, so duplicates should only exist when in DEV mode locally running it more than 1X per hour
# will drop duplicates at random to deal with this
log_df_for_email = log_df[log_df['timestamp'] >= past_24_hours.strftime('%Y-%m-%d %H:%M:%S')].sort_values(['trail', 'timestamp'], ascending=[True, True])[['trail', 'timestamp', 'trail_status', 'PRCP_4h', 'PRCP_8h', 'PRCP_16h', 'PRCP_24h', 'PRCP_48h', 'PRCP_72h', 'PRCP_120h', 'PRCP_168h']].drop_duplicates(subset=['trail', 'timestamp'])
log_df_visual = log_df_for_email.copy()

def reformat_timestamp_to_relative(timestamp, current_timestamp):
    """Reformat timestamp to 'Today XPM/XAM' or 'Yesterday XPM/XAM'."""
    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    if dt.date() == current_timestamp.date():
        return dt.strftime("Today %I%p")
    elif dt.date() == (current_timestamp - timedelta(days=1)).date():
        return dt.strftime("Yesterday %I%p")
    else:
        return dt.strftime('%m-%d %I%p')

log_df_for_email['timestamp'] = log_df_for_email['timestamp'].apply(lambda ts: reformat_timestamp_to_relative(ts, current_timestamp))
log_df_for_email['status_changed'] = log_df_for_email['trail_status'] != log_df_for_email['trail_status'].shift(1)
log_df_for_email = log_df_for_email[log_df_for_email['status_changed']]
log_df_for_email = log_df_for_email.drop(columns=['status_changed'])
print(log_df_for_email.head(50))

### REDUCE TOKEN SIZE

# Generate unique trail codes dynamically
unique_trails = log_df['trail'].dropna().unique()  # Exclude NaNs from unique values
trail_mapping = {trail: f"t_{chr(97 + i)}" for i, trail in enumerate(unique_trails)}

# Add a mapping for NaNs or unknowns
trail_mapping[None] = "t_unknown"
trail_mapping['Unknown Trail'] = "t_unknown"

# Reverse mapping for full text replacement
trail_reverse_mapping = {v: k for k, v in trail_mapping.items()}
print("trail_mapping:", trail_mapping)

# Generate unique status codes dynamically
unique_statuses = log_df['trail_status'].dropna().unique()  # Exclude NaNs from unique values
status_mapping = {status: f"s_{i + 1}" for i, status in enumerate(unique_statuses)}

# Add a mapping for NaNs or unknowns
status_mapping[None] = "s_unknown"
status_mapping['Unknown Status'] = "s_unknown"

status_reverse_mapping = {v: k for k, v in status_mapping.items()}
print("status_mapping:", status_mapping)

# Map trail and trail_status in the log_df_for_email DataFrame
log_df_for_email['trail'] = log_df_for_email['trail'].map(trail_mapping)
log_df_for_email['trail_status'] = log_df_for_email['trail_status'].map(status_mapping)

#####
## PLOT 
#####
# Define the color mapping
color_mapping = {
    'DEFINITE CLOSE': 'darkred',
    'LIKELY CLOSE': 'lightcoral',
    'LIKELY WET/OPEN': 'gold',
    'LIKELY OPEN': 'lightgreen',
    'DEFINITE OPEN': 'darkgreen'
}

# Apply the color mapping to the DataFrame
log_df_visual['color'] = log_df_visual['trail_status'].map(color_mapping).fillna('black')

# Create the plot
plt.figure(figsize=(12, 12))
log_df_visual = log_df_visual.sort_values(['timestamp', 'trail'], ascending = [True, False])
log_df_visual['timestamp'] = pd.to_datetime(log_df_visual['timestamp'])
log_df_visual['timestamp'] = log_df_visual['timestamp'].dt.strftime('%I%p')
scatter = plt.scatter(log_df_visual['timestamp'], log_df_visual['trail'], c=log_df_visual['color'], s=200, marker='s')  # 's' for squares, size 100

# Create a legend
legend_elements = [
    plt.Line2D([0], [0], marker='s', color='w', label='DEFINITE CLOSE', markersize=20, markerfacecolor='darkred'),
    plt.Line2D([0], [0], marker='s', color='w', label='LIKELY CLOSE', markersize=20, markerfacecolor='lightcoral'),
    plt.Line2D([0], [0], marker='s', color='w', label='LIKELY WET/OPEN', markersize=20, markerfacecolor='gold'),
    plt.Line2D([0], [0], marker='s', color='w', label='LIKELY OPEN', markersize=20, markerfacecolor='lightgreen'),
    plt.Line2D([0], [0], marker='s', color='w', label='DEFINITE OPEN', markersize=20, markerfacecolor='darkgreen'),
    plt.Line2D([0], [0], marker='s', color='w', label='Error (Uncommon)', markersize=20, markerfacecolor='black')
]


plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('P24 Hour Classifications')
today_date_time = datetime.today().strftime('%Y/%m/%d %I:%M %p')
plt.title(f"CORA Automated Trail Status Over Time: Last Updated ({today_date_time})", fontsize=18, pad=20)

plt.xticks(rotation=90)
plt.yticks(fontsize=12)
plt.savefig('trail_status_plot.png')  # Save the plot as an image file

filename_local = 'trail_status_plot_for_s3.png'
filename_s3 = 'cora_trail_status_plot.png'
plt.savefig(filename_local, dpi=300, bbox_inches='tight')  # Note the change here from fig.savefig to plt.savefig
s3.upload_file(filename_local, bucket_name, filename_s3, ExtraArgs={'ACL': 'public-read'})  # Uploading the second plot

print("Trail Status Image sent to S3")

plt.close()

#####
## JSON Writeout
#####

log_df_json_input = log_df.copy()
print("print len of log df for json processing", len(log_df_json_input))

# Function to get relevant timestamps for each group
def get_relevant_timestamps(group):
    # Sort by timestamp_with_date to ensure proper ordering
    group = group.sort_values(by='timestamp_with_date')

    # Detect status changes
    group['status_changed'] = group['trail_status'] != group['trail_status'].shift(1)

    # Get the most recent timestamp
    most_recent_timestamp = group.iloc[-1:]

    # Get the most recent timestamp where the trail_status is different from the current status
    if group['status_changed'].any():
        last_status_change = group[group['status_changed']].iloc[-2:-1]  # Select the second to last status change
    else:
        last_status_change = most_recent_timestamp

    # Add data_type column
    most_recent_timestamp['data_type'] = 'current'
    last_status_change['data_type'] = 'last_status_change'

    # Concatenate and clean up
    result = pd.concat([most_recent_timestamp, last_status_change]).drop_duplicates()
    result = result.sort_index()
    result = result.drop(columns=['status_changed'])
    
    return result

# Apply the function to each group
log_df_json = log_df_json_input.groupby('trail').apply(get_relevant_timestamps).reset_index(drop=True)
log_df_json['timestamp'] = log_df_json['timestamp'].apply(lambda ts: reformat_timestamp_to_relative(ts, current_timestamp))
print("filtered log data for JSON", log_df_json.head(30))

# Create a dictionary from the DataFrame
print(log_df_json.dtypes)
print("log_df_json", log_df_json.head(1))
log_df_json_dict = log_df_json.to_dict(orient='records')
print("Full Dict", log_df_json_dict)
log_df_json = json.dumps(log_df_json_dict, indent=4)
json_filename = 'trail_status_full.json'
with open(json_filename, 'w') as json_file:
    json_file.write(log_df_json)
s3.upload_file(json_filename, bucket_name, json_filename, ExtraArgs={'ACL': 'public-read'})

print("###############################")

# # Create a dictionary from the DataFrame
# log_df_json_trim = log_df_json[['trail', 'timestamp', 'timestamp_with_date', 'trail_status']].copy()
# print("log_df_json", log_df_json.head(1))
# log_df_json_dict = log_df_json.to_dict(orient='records')
# print("Trimmed Dict", log_df_json_dict)
# log_df_json = json.dumps(log_df_json_dict, indent=4)
# json_filename = 'trail_status_trimmed.json'
# with open(json_filename, 'w') as json_file:
#     json_file.write(log_df_json)
# s3.upload_file(json_filename, bucket_name, json_filename, ExtraArgs={'ACL': 'public-read'})

print("Trail Status JSON sent to S3")


######
# EMAIL
#######

# Check if the current hour is 8, 12, or 16 (4 PM)
current_hour = datetime.now().hour
if current_hour in [8,16]:

    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage

    # Load email config
    with open('creds/config_emails.json', 'r') as file:
        config = json.load(file)
        email_addresses = config['email_addresses']

    from_email = email_addresses[0]

    # Assuming final_df is your DataFrame
    # Filter DataFrame to include only 'trail' and 'trail_status' columns for the email
    df_filtered = final_df[['trail', 'trail_status']]
    df_filtered = df_filtered.sort_values(by='trail')


    # Set the flag for running OpenAI API
    run_openai_api = True  # Change this to False if you want to skip the OpenAI API call

    if run_openai_api:
        openai.api_key = openai_api_key

        # Call the OpenAI API to generate summary
        df_summary_input = ("""Please point out trail status changes and use the PRCP values to explain why you think the Trail status may have changed. Focus on the most recent changes (timestamp DESC) for each trail value: {}""".format(log_df_for_email.to_string(index=False)))

        response = openai.ChatCompletion.create(
            model="gpt-4o",  
            temperature=0.3,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": "You send out daily automated emails to all the local Cincinnati Mountain Bikers. You use a semi-casual semi-formal tone."},
                {"role": "user", "content": df_summary_input}
            ]
        )

        # Convert short codes back to full text in the response
        df_summary = response.choices[0].message.content
        for short_code, full_text in status_reverse_mapping.items():
            df_summary = df_summary.replace(short_code, full_text)
        for short_code, full_text in trail_reverse_mapping.items():
            df_summary = df_summary.replace(short_code, full_text)
        df_summary = df_summary.replace('\n', '<br>')
    else:
        df_summary = "OpenAI API was not called. Here is the trail status data without the summary."

    def format_trail_status(status):
        if status == "DEFINITE CLOSE":
            return f"<span style='color: darkred; font-weight: bold;'>{status}</span>"
        if status == "LIKELY CLOSE":
            return f"<span style='color: lightcoral; font-weight: bold;'>{status}</span>"
        elif status == "LIKELY WET/OPEN":
            return f"<span style='color: goldenrod; font-weight: bold;'>{status}</span>"
        elif status == "LIKELY OPEN":
            return f"<span style='color: lightgreen; font-weight: bold;'>{status}</span>"
        elif status == "DEFINITE OPEN":
            return f"<span style='color: darkgreen; font-weight: bold;'>{status}</span>"
        else:
            return f"<span style='color: black;'>{status}</span>"

    # # Create the HTML content for changes
    # def get_trail_changes(df, trail):
    #     changes = []
    #     for i in range(len(df) - 1):
    #         current_status = format_trail_status(status_reverse_mapping.get(df.iloc[i]['trail_status'], 'Unknown'))
    #         next_status = format_trail_status(status_reverse_mapping.get(df.iloc[i + 1]['trail_status'], 'Unknown'))
    #         timestamp = df.iloc[i + 1]['timestamp']
    #         changes.append(f"{current_status} → {next_status} ({timestamp})")
    #     return "<br>".join(changes)

    def get_trail_changes(df, trail):
        changes = []
        df = df.sort_values(by='timestamp', ascending=False)  # Ensure the DataFrame is sorted by timestamp in descending order
        for i in range(len(df) - 1):
            current_status = format_trail_status(status_reverse_mapping.get(df.iloc[i]['trail_status'], 'Unknown'))
            previous_status = format_trail_status(status_reverse_mapping.get(df.iloc[i + 1]['trail_status'], 'Unknown'))
            timestamp = df.iloc[i]['timestamp']
            changes.append(f"{previous_status} → {current_status} ({timestamp})")
        return "<br>".join(changes)  # Keep the order of changes as descending

    # trail_changes = log_df_for_email.groupby('trail').apply(lambda df: get_trail_changes(df, df['trail'].iloc[0])).to_dict()
    log_df_for_email = log_df_for_email.sort_values(['trail', 'timestamp'], ascending=[True, False])
    # trail_changes = dict(sorted(log_df_for_email.groupby('trail').apply(lambda df: get_trail_changes(df, df['trail'].iloc[0])).to_dict().items()))
    trail_changes = log_df_for_email.groupby('trail').apply(lambda df: get_trail_changes(df, df['trail'].iloc[0])).to_dict()
    trail_changes = dict(sorted(trail_changes.items()))

    bullet_points = df_filtered.apply(lambda row: f"<li><strong>{row['trail']}:</strong> {format_trail_status(row['trail_status'])}</li>", axis=1).tolist()
    bullet_points_html = "<ul>" + "".join(bullet_points) + "</ul>"

    # Prepare the email body with individual headers for each 'CORA Trail'
    today_date = datetime.now().strftime("%Y-%m-%d")
    email_body = f"""
    <h2>{today_date} Weather Report</h2>
    <h3>Trail Status Data:</h3>
    {bullet_points_html}
    <hr>
    </h2>OpenAI Analysis:</h2>
    {df_summary}
    </h2>Last Two Days - See Changes Over Time:</h2>
    """

    # for trail, changes in trail_changes.items():
    #     current_status = format_trail_status(status_reverse_mapping[log_df_for_email[log_df_for_email['trail'] == trail].iloc[0]['trail_status']])
    #     email_body += f"<h2>{trail_reverse_mapping[trail]}:</h2>"
    #     email_body += f"<h3>Current Status: {current_status}</h3>"
    #     email_body += f"<p>Changes:<br>{changes}</p>"

    email_body += "<img src='cid:trail_status_plot' alt='Trail Status Plot'>"

    # Creating the email message
    msg = MIMEMultipart('alternative')
    msg['From'] = from_email
    msg['To'] = ", ".join(email_addresses)
    msg['Subject'] = f'{today_date} CORA Trail Status Update'

    # Attach HTML content
    part = MIMEText(email_body, 'html')
    msg.attach(part)

    # Attach the image
    with open('trail_status_plot.png', 'rb') as img:
        img_data = img.read()
    img_part = MIMEImage(img_data, name='trail_status_plot.png')
    img_part.add_header('Content-ID', '<trail_status_plot>')
    msg.attach(img_part)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:  # Use the correct SMTP server and port
        server.starttls()
        server.login(from_email, gmail_api_key)   
        server.sendmail(from_email, email_addresses, msg.as_string())
    print("Email sent successfully!")

else:
    print("Current hour is not 8AM or 4PM, email not sent")
