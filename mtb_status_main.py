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

# config_file_path = 'data/dev_config.json'
config_file_path = '/root/cora_trail_status_classifier/data/prod_config.json'

with open(config_file_path, 'r') as config_file:
    config = json.load(config_file)

data_directory = config['paths']['data_directory']
os.chdir(data_directory)

# lookback_days_list = [2, 3, 4, 5, 6, 7, 11, 18]

max_temp_freeze_thaw = 32

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

    # Filter data to the most recent 15 days
    days_ago_filter = datetime.now() - timedelta(days=15)
    hourly_weather['HOUR'] = pd.to_datetime(hourly_weather['HOUR'])
    hourly_weather = hourly_weather[hourly_weather['HOUR'] >= days_ago_filter]

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

def calculate_freeze_thaw_points(temperature):
    if temperature <= 27:
        return 10/20
    elif temperature <= 29:
        return 5/20
    elif temperature <= 30:
        return 2.5/20
    elif temperature <= max_temp_freeze_thaw: # this should be 32, adding as a variable just for dev/test purposes
        return 120/20
    elif temperature > max_temp_freeze_thaw: # use this negative as a helper for later function to cumulative sum
        return -1
    return 0

print("LINE 204: hourly weather head", hourly_accuweather.head(5))
hourly_accuweather['freeze_thaw_points'] = hourly_accuweather['MAX TEMPERATURE'].apply(calculate_freeze_thaw_points)

if 'freeze_thaw_points' in hourly_accuweather.columns:
    print("freeze_thaw_points exists")
else:
    assert False, "freeze_thaw_points column is missing"

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

column_names = {'Trail': 'trail', 'DATE': 'DATE', 'MAX TEMPERATURE': 'TMAX', 'TOTAL PRECIPITATION': 'PRCP', 
'DEW_POINT': 'DEW_POINT', 'PRECIPITATION PROBABILITY': 'PROB_ADJ', 'freeze_thaw_points': 'freeze_thaw_points'} # 'SNOW_FLAG': 'SNOW_FLAG'}
weather_sorted = weather_sorted.reset_index().rename(columns=column_names)
print(weather_sorted.head(5))
weather_data_main = weather_sorted.copy()
print("---------------")

weather_data_main = weather_data_main.sort_values(['trail','DATE'])
weather_data_main.to_csv("weather_data_main.csv", index=False)

# Define the local and S3 filenames
filename_local = 'weather_data_main.csv'
filename_s3 = 'weather_data_main.csv'
s3.upload_file(filename_local, bucket_name, filename_s3, ExtraArgs={'ACL': 'public-read'})
print(f"File uploaded to S3: s3://{bucket_name}/{filename_s3}")

### ROLLING METRICS CALCULATED

weather_data_hourly = weather_data_main[weather_data_main["HOUR"].notnull()]
# weather_data_daily = weather_data_main[weather_data_main["HOUR"].isnull()]

yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)

print("Hourly Data PRINT HEAD LINE 271:\n", weather_data_hourly.sort_values(["trail", "DATE"], ascending = [False, False]).head(15))
lookback_hours_list = [4, 8, 16, 24, 48, 72, 96, 120, 144, 168, 336]

# Assume weather_data_hourly and weather_data_daily are already defined
# daily_df = calculate_rolling_metrics_daily(weather_data_daily, lookback_days_list)


hourly_df = calculate_rolling_metrics_hourly(weather_data_hourly, lookback_hours_list)

if 'freeze_thaw_points_cumulative' in hourly_df.columns:
    print("freeze_thaw_points_cumulative exists")
else:
    assert False, "freeze_thaw_points_cumulative column is missing"

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
                     'PRCP_24h', 'PRCP_48h', 'PRCP_72h', 'PRCP_120h', 'PRCP_168h', 'PRCP_336h', 
                     'PRCP_4h', 'PRCP_8h', 'PRCP_16h',
                     'TMAX_24h', 'TMAX_48h', 'TMAX_72h', 'TMAX_120h', 'TMAX_168h', 'TMAX_336h', 
                     'TMAX_4h', 'TMAX_8h', 'TMAX_16h',
                     'DEW_POINT_24h', 'DEW_POINT_48h', 'DEW_POINT_72h', 'DEW_POINT_120h', 'DEW_POINT_168h', 'DEW_POINT_336h',
                     'DEW_POINT_4h', 'DEW_POINT_8h', 'DEW_POINT_16h',
                     'freeze_thaw_points', 'freeze_thaw_points_cumulative'
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

def adjust_prcp(prcp, tmax, dew_point, trail, prcp_336h):
    # Adjust based on TMAX
    if tmax > 95:
        prcp *= 0.90
    elif tmax > 90:
        prcp *= 0.95   
    elif tmax > 85:
        prcp *= 1   
    elif tmax > 70:
        prcp *= 1 
    elif tmax <= 38:
        prcp *= 1.5
    elif tmax <= 45:
        prcp *= 1.35
    elif tmax <= 55:
        prcp *= 1.15    
    elif tmax <= 65:
        prcp *= 1.05 

    dew_point_temp_diff = tmax - dew_point

    # Adjust based on DEW_POINT DIFF
    if dew_point_temp_diff < 5:
        prcp *= 1.25
    elif dew_point_temp_diff < 10:
        prcp *= 1.15
    elif dew_point_temp_diff < 15:
        prcp *= 1.05  
    elif dew_point_temp_diff < 20:
        prcp *= 0.95
    elif dew_point_temp_diff < 25:
        prcp *= 0.9 
    elif dew_point_temp_diff < 35:
        prcp *= 0.85

    ## If its been super dry past two weeks, adjust PRCP values
    if prcp_336h > 12:
        prcp *= 1.4
    elif prcp_336h > 10:
        prcp *= 1.3 
    elif prcp_336h > 8.5:
        prcp *= 1.2
    elif prcp_336h > 6.5:
        prcp *= 1.1
    elif prcp_336h > 5:
        prcp *= 1.0 
    elif prcp_336h <= 5:
        prcp *= 0.95
    elif prcp_336h <= 2.5:
        prcp *= 0.5
    
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
    print(f"\nTrail: {row['trail']}")
    print(f"Freeze/thaw points: {row['freeze_thaw_points']}")
    print(f"Cumulative freeze/thaw points: {row['freeze_thaw_points_cumulative']}")
    
    if row['freeze_thaw_points_cumulative'] >= 1:  # 1 represents 20/20 points
        print(f"FREEZE/THAW condition met for {row['trail']} with {row['freeze_thaw_points_cumulative']} points")
        return pd.Series({
            'trail_status': 'FREEZE/THAW', 
            'next_closest_trail_status': 'LIKELY CLOSE', 
            'weighted_avg_score': 999
        })
    prcp_4h = row['PRCP_4h']
    prcp_8h = row['PRCP_8h']
    prcp_16h = row['PRCP_16h']
    prcp_24h = row['PRCP_24h']
    prcp_48h = row['PRCP_48h']
    prcp_72h = row['PRCP_72h']
    prcp_120h = row['PRCP_120h']
    prcp_168h = row['PRCP_168h']
    prcp_336h = row['PRCP_336h']
    tmax_4h = row['TMAX_4h']
    tmax_8h = row['TMAX_8h']
    tmax_16h = row['TMAX_16h']
    tmax_24h = row['TMAX_24h']
    tmax_48h = row['TMAX_48h']
    tmax_72h = row['TMAX_72h']
    tmax_120h = row['TMAX_120h']
    tmax_168h = row['TMAX_168h']
    tmax_336h = row['TMAX_336h']
    dew_point_4h = row['DEW_POINT_4h']
    dew_point_8h = row['DEW_POINT_8h']
    dew_point_16h = row['DEW_POINT_16h']
    dew_point_24h = row['DEW_POINT_24h']
    dew_point_48h = row['DEW_POINT_48h']
    dew_point_72h = row['DEW_POINT_72h']
    dew_point_120h = row['DEW_POINT_120h']
    dew_point_168h = row['DEW_POINT_168h']
    dew_point_336h = row['DEW_POINT_336h']
    trail = row['trail']

    # Adjust precipitation values
    prcp_4h = adjust_prcp(prcp_4h, tmax_4h, dew_point_4h, trail, prcp_336h)
    prcp_8h = adjust_prcp(prcp_8h, tmax_8h, dew_point_8h, trail, prcp_336h)
    prcp_16h = adjust_prcp(prcp_16h, tmax_16h, dew_point_16h, trail, prcp_336h)
    prcp_24h = adjust_prcp(prcp_24h, tmax_24h, dew_point_24h, trail, prcp_336h)
    prcp_48h = adjust_prcp(prcp_48h, tmax_48h, dew_point_48h, trail, prcp_336h)
    prcp_72h = adjust_prcp(prcp_72h, tmax_72h, dew_point_72h, trail, prcp_336h)
    prcp_120h = adjust_prcp(prcp_120h, tmax_120h, dew_point_120h, trail, prcp_336h)
    prcp_168h = adjust_prcp(prcp_168h, tmax_168h, dew_point_168h, trail, prcp_336h)
    prcp_336h = adjust_prcp(prcp_336h, tmax_336h, dew_point_336h, trail, prcp_336h)

    dimensions = ['prcp_4h', 'prcp_8h', 'prcp_16h', 'prcp_24h', 'prcp_48h', 'prcp_72h', 'prcp_120h', 'prcp_168h', 'prcp_336h']
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
                    count += 0.4
                if greater_than['prcp_8h'][status] <= prcp_values['prcp_8h'] <= less_than['prcp_8h'][status]:
                    count += 0.3
                if greater_than['prcp_16h'][status] <= prcp_values['prcp_16h'] <= less_than['prcp_16h'][status]:
                    count += 0.2
                if greater_than['prcp_24h'][status] <= prcp_values['prcp_24h'] <= less_than['prcp_24h'][status]:
                    count += 0.75
                if greater_than['prcp_48h'][status] <= prcp_values['prcp_48h'] <= less_than['prcp_48h'][status]:
                    count += 1.20
                if greater_than['prcp_72h'][status] <= prcp_values['prcp_72h'] <= less_than['prcp_72h'][status]:
                    count += 1.10
                if greater_than['prcp_120h'][status] <= prcp_values['prcp_120h'] <= less_than['prcp_120h'][status]:
                    count += 1.00
                if greater_than['prcp_168h'][status] <= prcp_values['prcp_168h'] <= less_than['prcp_168h'][status]:
                    count += 0.90
                if greater_than['prcp_336h'][status] <= prcp_values['prcp_336h'] <= less_than['prcp_336h'][status]:
                    count += 0.80
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

        return closest_status, next_closest_status, weighted_average_score
    
    prcp_values = {
        'prcp_4h': prcp_4h,
        'prcp_8h': prcp_8h,
        'prcp_16h': prcp_16h,
        'prcp_24h': prcp_24h,
        'prcp_48h': prcp_48h,
        'prcp_72h': prcp_72h,
        'prcp_120h': prcp_120h,
        'prcp_168h': prcp_168h,
        'prcp_336h': prcp_336h
    }
    print("processing this row of data", row[['trail', 'PRCP_4h', 'PRCP_8h', 'PRCP_16h', 'PRCP_24h', 'PRCP_48h', 'PRCP_72h', 'PRCP_120h', 'PRCP_168h', 'PRCP_336h']])
    print("Result: ", get_trail_status(prcp_values))
    
    closest_status, next_closest_status, weighted_avg_score = get_trail_status(prcp_values)
    return pd.Series({'trail_status': closest_status, 'next_closest_trail_status': next_closest_status, 'weighted_avg_score': round(weighted_avg_score, 2)})

# Applying the updated function to the DataFrame
final_df[['trail_status', 'next_closest_trail_status', 'weighted_avg_score']] = final_df.apply(trail_status, axis=1)
print(final_df[['trail', 'PRCP_24h', 'PRCP_8h', 'PRCP_48h', 'PRCP_16h', 'PRCP_72h', 'PRCP_120h', 'PRCP_168h', 'PRCP_336h', 'TMAX_24h', 'DEW_POINT_24h', 'trail_status', 'next_closest_trail_status', 'weighted_avg_score']])


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
        return pd.DataFrame(columns=['trail', 'PRCP_4h', 'PRCP_8h', 'PRCP_16h', 'PRCP_24h', 'PRCP_48h', 'PRCP_72h', 'PRCP_120h', 'PRCP_168h', 'PRCP_336h', 'trail_status', 'next_closest_trail_status', 'weighted_avg_score', 'timestamp'])

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

# # TEMP NIGHTHAWK ADD 2024-08-17

# # Find the row where Trail is 'Nighthawk Bike Park'
# nighthawk_row = final_df[final_df['trail'] == 'Nighthawk Bike Park']
# duplicated_row = nighthawk_row.copy()
# duplicated_row['trail_status'] = 'LIKELY CLOSE'
# duplicated_row['next_closest_trail_status'] = 'LIKELY WET/OPEN'
# # Append the duplicated and modified row back to the DataFrame
# final_df = pd.concat([final_df, duplicated_row], ignore_index=True)

log_df = append_to_log(final_df[['trail', 'PRCP_4h', 'PRCP_8h', 'PRCP_16h', 'PRCP_24h', 'PRCP_48h', 'PRCP_72h', 'PRCP_120h', 'PRCP_168h', 'PRCP_336h', 'trail_status', 'next_closest_trail_status', 'weighted_avg_score']]) #timestamp is created, not inputted



###
# 2024-08-07 temp solution
# I do need to figure out a better way to handle "new variables" to the logic, when I dont have their history stored
log_df['PRCP_336h'] = log_df['PRCP_336h'].fillna(5) # 5 inches corresponds with YELLOW

# To send full timestamp for QA purposes to Nathan at CORA
log_df['timestamp_with_date'] = log_df['timestamp']

print("LOG DF", log_df.sort_values('timestamp', ascending = False).head(50))
#### VIEW LOG

#### VIEW LOG
print("VIEW LOG ######")
# Get current timestamp and past 24 hours timestamp
current_timestamp = datetime.now().replace(minute=59, second=0, microsecond=0)

time_filter = current_timestamp - timedelta(hours=336)

# Filter log_df based on timestamp in past 24 hours and sort it
# the script runs once per hour, so duplicates should only exist when in DEV mode locally running it more than 1X per hour
# will drop duplicates at random to deal with this
log_df_for_email = log_df[log_df['timestamp'] >= time_filter.strftime('%Y-%m-%d %H:%M:%S')].sort_values(['trail', 'timestamp'], ascending=[True, True])[['trail', 'timestamp', 'trail_status', 'PRCP_4h', 'PRCP_8h', 'PRCP_16h', 'PRCP_24h', 'PRCP_48h', 'PRCP_72h', 'PRCP_120h', 'PRCP_168h', 'PRCP_336h']].drop_duplicates(subset=['trail', 'timestamp'])
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

# Generate unique status codes dynamically
unique_statuses = log_df['trail_status'].dropna().unique()  # Exclude NaNs from unique values
status_mapping = {status: f"s_{i + 1}" for i, status in enumerate(unique_statuses)}
# Add a mapping for NaNs or unknowns
status_mapping[None] = "s_unknown"
status_mapping['Unknown Status'] = "s_unknown"
status_reverse_mapping = {v: k for k, v in status_mapping.items()}

# print("CHECK TO MAKE SURE ALL TRAILS IN LOG_DF_FOR_EMAIL")
# print(log_df_for_email['trail'].unique())

# Map trail and trail_status in the log_df_for_email DataFrame
log_df_for_email['trail'] = log_df_for_email['trail'].map(trail_mapping)
log_df_for_email['trail_status'] = log_df_for_email['trail_status'].map(status_mapping)

# print("CHECK TO MAKE SURE ALL TRAILS IN LOG_DF_FOR_EMAIL")
# print(log_df_for_email['trail'].unique())
#####
## PLOT 
#####
# Define the color mapping
color_mapping = {
    'FREEZE/THAW': 'skyblue',
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
    plt.Line2D([0], [0], marker='s', color='w', label='FREEZE/THAW', markersize=20, markerfacecolor='skyblue'),
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

def get_relevant_timestamps(group):
    # Sort by timestamp_with_date to ensure proper ordering
    group['timestamp_with_date'] = pd.to_datetime(group['timestamp_with_date'])
    group = group.sort_values(by='timestamp_with_date')

    # Detect status changes
    group['status_changed'] = group['trail_status'] != group['trail_status'].shift(1)

    # print("group head", group.sort_values('timestamp', ascending = False).head(25))

    # Get the most recent timestamp
    most_recent_timestamp = group.iloc[-1:]

    # Get the most recent timestamp where the trail_status is different from the current status
    if group['status_changed'].any():
        # Get the last status change row
        last_status_change = group[group['status_changed']].tail(1)
    else:
        last_status_change = most_recent_timestamp

    # Find the timestamp 1 hour prior to the last status change
    # 2024-08-07: for some reason, the script failed for a few hours so one_hour_prior was also failing
    one_hour_prior = last_status_change['timestamp_with_date'].iloc[0] - pd.Timedelta(hours=1)

    # Find the row with the timestamp 1 hour prior
    one_hour_prior_row = group[group['timestamp_with_date'] == one_hour_prior]

    if one_hour_prior_row.empty:
        # If no exact match is found, find the closest prior timestamp
        # This works properly because the DATETIME values are sorted DESCENDING
        one_hour_prior_row = group[group['timestamp_with_date'] < one_hour_prior].tail(1)
    
    print(group['trail'].unique())
    print("Most recent timestamp: ", most_recent_timestamp['trail_status'].values[0])
    print("1 hour prior timestamp: ", one_hour_prior_row['trail_status'].values[0])

    assert most_recent_timestamp['trail_status'].values[0] != one_hour_prior_row['trail_status'].values[0]

    # Add data_type column
    most_recent_timestamp['data_type'] = 'current_trail_status'
    one_hour_prior_row['data_type'] = 'previous_trail_status'

    # Concatenate and clean up
    # print("MOST RECENT", most_recent_timestamp)
    # print("LAST STATUS CHANGE", one_hour_prior_row)
    result = pd.concat([most_recent_timestamp, one_hour_prior_row]).drop_duplicates()
    result = result.sort_index()
    result = result.drop(columns=['status_changed'])

    return result

# Apply the function to each group
log_df_json = log_df_json_input.groupby('trail').apply(get_relevant_timestamps).reset_index(drop=True)
log_df_json['timestamp'] = log_df_json['timestamp'].apply(lambda ts: reformat_timestamp_to_relative(ts, current_timestamp))
log_df_json['timestamp_with_date'] = log_df_json['timestamp_with_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
log_df_json['weighted_avg_score'] = log_df_json['weighted_avg_score'].fillna("N/A")
print("filtered log data for JSON", log_df_json.head(5))

# Create a dictionary from the DataFrame
# print(log_df_json.dtypes)
print("log_df_json", log_df_json.head(1))
log_df_json_dict = log_df_json.to_dict(orient='records')
# print("Full Dict", log_df_json_dict)
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

current_hour = datetime.now().hour
if current_hour in [1000000]: # purposefully disabling this for now
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
        print("CHECK TO MAKE SURE ALL TRAILS IN LOG_DF_FOR_EMAIL")
        print(log_df_for_email['trail'].unique())
        df_summary_input = ("""Please point out trail status changes and use the PRCP values to explain why you think the Trail status may have changed. Please convert hours to days when 24h or higher (e.g. 24h is 1 day, 48h is 2 days):  {}""".format(log_df_for_email.to_string(index=False))) #Focus on the most recent changes (timestamp DESC) for each trail value:

        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06",  
            temperature=0.2,
            max_tokens=3000,
            messages=[
                {"role": "system", "content": "You send out daily automated emails to all the local Cincinnati Mountain Bikers. Do not output any ** characters in your email. Leave the t_1, t_2, etc formatting as is, do not try to reformat. You use a semi-formal but laid back tone. You begin your emails with an analysis summary, then go trail by trail giving more detailed analysis. You order your responses t_1, t_2, t_3, and so on."},
                {"role": "user", "content": df_summary_input}
            ]
                        # response_format=EmailOutput
        #                 from pydantic import BaseModel
        # class EmailOutput(BaseModel):
        #     trail: str
        #     explanation: str
        )

        # Convert short codes back to full text in the response
        df_summary = response.choices[0].message.content
        for short_code, full_text in status_reverse_mapping.items():
            df_summary = df_summary.replace(short_code, full_text)
        for short_code, full_text in trail_reverse_mapping.items():
            df_summary = df_summary.replace(short_code, full_text)
        df_summary = df_summary.replace('\n', '<br>')
        df_summary = df_summary.replace('**', '')
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

    bullet_points = df_filtered.apply(lambda row: f"<li>{row['trail']}: {format_trail_status(row['trail_status'])}</li>", axis=1).tolist()
    bullet_points_html = "<ul>" + "".join(bullet_points) + "</ul>"


    # Prepare the email body with individual headers for each 'CORA Trail'
    today_date = datetime.now().strftime("%Y-%m-%d")
    email_body = f"""
    <h2>{today_date} Cincinnati MTB Trail Report</h2>
    <h3>Trail Status Data:</h3>
    {bullet_points_html}
    <hr>
    </h2>OpenAI Analysis:</h2>
    <p>{df_summary}</p>
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
