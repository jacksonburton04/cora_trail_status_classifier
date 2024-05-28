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

warnings.filterwarnings('ignore')

config_file_path = 'data/dev_config.json'
# config_file_path = '/root/cora_trail_status_classifier/data/prod_config.json'

with open(config_file_path, 'r') as config_file:
    config = json.load(config_file)

data_directory = config['paths']['data_directory']
os.chdir(data_directory)

lookback_days_list = [2, 3, 4, 5, 6, 7, 11, 18]

s3_client = boto3.client('s3')
s3 = boto3.client('s3')
bucket_name = 'mtb-trail-condition-predictions'

### FUNCTIONS

def get_weather_data(lat, lon, date, api_key):
    base_url = "https://api.openweathermap.org/data/3.0/onecall/day_summary"
    params = {
        'lat': lat,
        'lon': lon,
        'date': date,
        'appid': api_key,
        'units': 'imperial'  # For Fahrenheit
    }

    response = requests.get(base_url, params=params)
    return response.json()

def calculate_dew_point(T, RH):
    """
    Calculate the dew point temperature given the temperature and relative humidity
    using the Magnus formula.

    Parameters:
    T (float): the temperature in degrees Fahrenheit
    RH (float): the relative humidity in percent

    Returns:
    float: the dew point temperature in degrees Fahrenheit
    """
    # Convert temperature from Fahrenheit to Celsius
    T_c = (T - 32) * 5.0/9.0
    
    # Constants for the Magnus formula
    a = 17.27
    b = 237.7
    
    # Magnus formula for dew point in Celsius
    alpha = ((a * T_c) / (b + T_c)) + math.log(RH/100.0)
    dew_point_c = (b * alpha) / (a - alpha)
    
    # Convert dew point back to Fahrenheit
    dew_point_f = (dew_point_c * 9.0/5.0) + 32
    
    return dew_point_f

import math


### APIS (ENCRYPTED) 

## OPEN WEATHER API
from cryptography.fernet import Fernet
with open("creds/encryption_openweather_key.txt", "rb") as f:
    encryption_key = f.read()
cipher_suite = Fernet(encryption_key)
with open("creds/encrypted_openweather_api_key.txt", "rb") as f:
    encrypted_api_key = f.read()
decrypted_api_key = cipher_suite.decrypt(encrypted_api_key)
api_key = decrypted_api_key.decode()

## GMAIL API
with open("creds/encryption_gmail_key.txt", "rb") as f:
    encryption_key = f.read()
cipher_suite = Fernet(encryption_key)
with open("creds/encrypted_gmail_api_key.txt", "rb") as f:
    encrypted_api_key = f.read()
decrypted_api_key = cipher_suite.decrypt(encrypted_api_key)
gmail_api_key = decrypted_api_key.decode()

## CHATGPT API
with open("creds/encryption_openai_key.txt", "rb") as f:
    encryption_key = f.read()
cipher_suite = Fernet(encryption_key)
with open("creds/encrypted_openai_api_key.txt", "rb") as f:
    encrypted_api_key = f.read()
decrypted_api_key = cipher_suite.decrypt(encrypted_api_key)
openai_api_key = decrypted_api_key.decode()


file_key = 'data/trail_locations.csv'
obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
print("Reading in file from S3 bucket")
df_trail_locations = pd.read_csv(obj['Body'])
print(df_trail_locations.head(5))

exclude = "minutely,hourly,alerts"
pickle_file = 'weather_data.pickle'

#######
pickle_file = 'hourly_weather_data.pickle'

print("Fetching new hourly data for:", datetime.now().date())

# Check if the pickle file has been modified in the past hour
if os.path.exists(pickle_file):
    file_mod_time = datetime.fromtimestamp(os.path.getmtime(pickle_file))
    if datetime.now() - file_mod_time < timedelta(minutes=50):
        print("Pickle file has been modified in the past hour. Loading existing data.")
        with open(pickle_file, 'rb') as f:
            hourly_weather = pickle.load(f)
    else:
        print("Pickle file is older than X hours. Fetching new data.")
        hourly_weather = pd.DataFrame()
else:
    hourly_weather = pd.DataFrame()

# TEMPORARY 05-25 solution
# hourly_weather = pd.read_csv("temp_hour_df_05_25.csv")    

# Define the time range for the past two days
start_date = datetime.now() - timedelta(days=2)
end_date = datetime.now()

# Loop through each trail location
for index, row in df_trail_locations.iterrows():
    lat = row['Latitude']
    lon = row['Longitude']
    trail = row['Trail']
    url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,daily,alerts&appid={api_key}&units=imperial"
    response = requests.get(url).json()
    hourly_forecast = response['hourly']

    # Extract hourly precipitation data
    new_data = []
    for hour in hourly_forecast:
        hour_time = datetime.fromtimestamp(hour['dt'])
        if start_date <= hour_time <= end_date:
            precip_inches = 0
            if 'rain' in hour and '1h' in hour['rain']:
                precip_inches += (hour['rain']['1h'] / 25.4)
            if 'snow' in hour and '1h' in hour['snow']:
                precip_inches += (hour['snow']['1h'] / 25.4)
            humidity = hour['humidity']
            temp = hour['temp']
            dew_point = calculate_dew_point(temp, humidity)
            # Append data to the new_data list
            new_data.append([trail, hour_time.strftime('%Y-%m-%d %H:%M'), precip_inches, temp, dew_point]) #snow_flag

    new_hourly_data = pd.DataFrame(new_data, columns=['Trail', 'HOUR', 'TOTAL PRECIPITATION', 'MAX TEMPERATURE', 'DEW_POINT'])  
    hourly_weather = pd.concat([new_hourly_data, hourly_weather]).drop_duplicates(['Trail','HOUR']).reset_index(drop=True)

# Save to pickle file
with open(pickle_file, 'wb') as f:
    pickle.dump(hourly_weather, f)
print("Hourly data updated and saved to pickle file.")

# Filter for the past two days
hourly_weather['HOUR'] = pd.to_datetime(hourly_weather['HOUR'])
hourly_weather['DATE'] = hourly_weather['HOUR'].dt.strftime('%Y-%m-%d')
hourly_weather['HOUR'] = hourly_weather['HOUR'].dt.strftime('%H')
hourly_weather = hourly_weather[(pd.to_datetime(hourly_weather['DATE']) >= start_date) & (pd.to_datetime(hourly_weather['DATE']) <= end_date)]

print(hourly_weather.sort_values(["DATE","HOUR"], ascending = [False, False]).head(20))

########
# END
#########


pickle_file = 'historical_one_week_all_trails.pickle'

if os.path.exists(pickle_file) and datetime.fromtimestamp(os.path.getmtime(pickle_file)).date() == datetime.now().date():
    with open(pickle_file, 'rb') as f:
        historical_one_week_all_trails = pickle.load(f)
    print("Already have today's historical data. Loading from pickle file.")
else:
    # Initialize an empty DataFrame
    historical_one_week_all_trails = pd.DataFrame()

    # Loop through each trail in df_trail_locations
    for index, row in df_trail_locations.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        trail = row['Trail']
        data = []
        # Loop through last X days
        for i in range(1, 10): 
            date_var = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            response = get_weather_data(lat, lon, date_var, api_key)
            try:
                temp_max = response['temperature']['max']
                precipitation = response['precipitation']['total'] * 0.0393701
                dew_point = calculate_dew_point(response['temperature']['afternoon'], response['humidity']['afternoon'])
                data.append([trail, date_var, temp_max, precipitation, dew_point]) #snow_flag
            except KeyError as e:
                # Fetch a default error message from the response, if available, otherwise use a specific KeyError message
                error_msg = response.get('message', f'KeyError for {e}')
                print(f"Error for {trail} on {date_var}: {error_msg}")

        historical_data = pd.DataFrame(data, columns=['Trail', 'DATE', 'MAX TEMPERATURE', 'TOTAL PRECIPITATION', 'DEW_POINT']) #'SNOW_FLAG'])
        historical_one_week_all_trails = pd.concat([historical_one_week_all_trails, historical_data])

    # Save to pickle
    with open(pickle_file, 'wb') as f:
        pickle.dump(historical_one_week_all_trails, f)
    print("Did not have today's historical data, pulling API data and saving to pickle file.")

historical_one_week_all_trails.to_csv("historical_one_week_all_trails.csv")
hourly_weather.to_csv("hourly_weather.csv")


yesterday = datetime.now() - timedelta(days=1)
yesterday_date_str = yesterday.strftime('%Y-%m-%d')
historical_one_week_all_trails['DATE'] = pd.to_datetime(historical_one_week_all_trails['DATE'], format='%Y-%m-%d')
historical_one_week_all_trails_filtered = historical_one_week_all_trails[historical_one_week_all_trails['DATE'] < yesterday]

# COMBINE WEATHER DFs
weather_append = pd.concat([hourly_weather, historical_one_week_all_trails_filtered])
weather_append = weather_append.drop_duplicates(subset=['DATE', 'Trail', 'HOUR', 'TOTAL PRECIPITATION'], keep='first')
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
weather_data_daily = weather_data_main[weather_data_main["HOUR"].isnull()]

yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
# Filter data for yesterday's date and group by DATE
weather_data_daily_append = weather_data_main[
    weather_data_main['DATE'] == yesterday
].groupby(['DATE','trail']).agg({
    'PRCP': 'sum',
    'TMAX': 'max',
    'DEW_POINT': 'max'
}).reset_index()
weather_data_daily = pd.concat([weather_data_daily, weather_data_daily_append], ignore_index=True)

print("Hourly Data:\n", weather_data_hourly.head())
print("Daily Data:\n", weather_data_daily.head())
print("New Daily Data to Append:\n", weather_data_daily_append.head())
print("Updated Daily Data:\n", weather_data_daily.head())


def calculate_rolling_metrics_daily(daily_df, lookback_days_list):
    """
    Calculates rolling sums and averages for specified columns over a range of lookback days for daily data.

    Parameters:
    - daily_df: DataFrame containing the daily data.
    - lookback_days_list: List of integers representing the lookback days for rolling calculations.

    Returns:
    - daily_df: DataFrame with added columns for rolling calculations.
    """
    daily_df['DATE'] = pd.to_datetime(daily_df['DATE'])
    daily_df = daily_df.sort_values(['trail', 'DATE']) # this is crucial to the logic below

    # Add rolling metrics for past X days
    for i in lookback_days_list:
        for col in ['PRCP', 'TMAX', 'DEW_POINT']:  # Columns to calculate rolling metrics for
            if col in daily_df.columns:  # Check if column exists in DataFrame
                new_col_name = f'{col}_{i}d'
                if col == 'PRCP':  
                    daily_df[new_col_name] = daily_df[col].rolling(window=i, min_periods=1).sum()
                elif col == 'DEW_POINT':  
                    daily_df[new_col_name] = daily_df[col].rolling(window=i, min_periods=1).mean()
                else:  #
                    daily_df[new_col_name] = daily_df[col].rolling(window=i, min_periods=1).max()

    daily_df.fillna(0, inplace=True)
    return daily_df

def calculate_rolling_metrics_hourly(hourly_df, lookback_hours_list):
    """
    Calculates rolling sums and averages for specified columns over a range of lookback hours for hourly data.

    Parameters:
    - hourly_df: DataFrame containing the hourly data.
    - lookback_hours_list: List of integers representing the lookback hours for rolling calculations.

    Returns:
    - hourly_df: DataFrame with added columns for rolling calculations.
    """
   # Ensure the 'DATE' and 'HOUR' columns are in datetime format and sort the DataFrame by 'DATE' and 'HOUR'
    hourly_df['DATE'] = pd.to_datetime(hourly_df['DATE'])
    hourly_df['HOUR'] = hourly_df['HOUR'].astype(int)  # Convert HOUR to integer
    hourly_df['DATETIME'] = hourly_df.apply(lambda row: pd.Timestamp(row['DATE']) + pd.to_timedelta(row['HOUR'], unit='h'), axis=1)
    hourly_df = hourly_df.sort_values(['trail', 'DATETIME']) # this is crucial to the logic below

    for i in lookback_hours_list:
        for col in ['PRCP', 'TMAX', 'DEW_POINT']: 
            if col in hourly_df.columns:  
                new_col_name = f'{col}_{i}h'
                if col == 'PRCP': 
                    hourly_df[new_col_name] = hourly_df[col].rolling(window=i, min_periods=1).sum()
                elif col == 'DEW_POINT': 
                    hourly_df[new_col_name] = hourly_df[col].rolling(window=i, min_periods=1).mean()
                else:  
                    hourly_df[new_col_name] = hourly_df[col].rolling(window=i, min_periods=1).max()

    hourly_df.fillna(0, inplace=True)
    return hourly_df

def combine_hourly_and_daily(hourly_df, daily_df):
    """
    Combines hourly and daily data into a cumulative DataFrame.

    Parameters:
    - hourly_df: DataFrame containing the hourly data with rolling metrics.
    - daily_df: DataFrame containing the daily data with rolling metrics.

    Returns:
    - combined_df: Cumulative DataFrame.
    """
    # Drop 'DATETIME' column from hourly_df to avoid confusion
    hourly_df.drop(columns=['DATETIME'], inplace=True)

    # Concatenate the hourly and daily data
    combined_df = pd.concat([daily_df, hourly_df], ignore_index=True)

    return combined_df

# Example usage:
lookback_days_list = [1, 2, 3, 5, 7, 10]
lookback_hours_list = [4, 8, 16]

# Assume weather_data_hourly and weather_data_daily are already defined
daily_df = calculate_rolling_metrics_daily(weather_data_daily, lookback_days_list)
hourly_df = calculate_rolling_metrics_hourly(weather_data_hourly, lookback_hours_list)

# daily_df_trim = daily_df.groupby('trail', as_index=False).apply(lambda x: x.loc[x['DATE'].idxmax()]).reset_index(drop=True)
daily_df_trim = daily_df.sort_values(by='DATE', ascending=False).drop_duplicates(subset = 'trail').reset_index(drop=True)
hourly_df_trim = hourly_df.sort_values(by=['DATE', 'HOUR'], ascending=[False, False]).drop_duplicates(subset = 'trail').reset_index(drop=True)
final_df = pd.merge(daily_df_trim, hourly_df_trim, on='trail')
final_df.to_csv("final_df.csv")

### 

df_class = final_df[['trail', 
                     'PRCP_1d', 'PRCP_2d', 'PRCP_3d', 'PRCP_5d', 'PRCP_7d', 'PRCP_10d', 
                     'PRCP_4h', 'PRCP_8h', 'PRCP_16h',
                     'TMAX_1d', 'TMAX_2d', 'TMAX_3d', 'TMAX_5d', 'TMAX_7d', 'TMAX_10d', 
                     'TMAX_4h', 'TMAX_8h', 'TMAX_16h',
                     'DEW_POINT_1d', 'DEW_POINT_2d', 'DEW_POINT_3d', 'DEW_POINT_5d', 'DEW_POINT_7d', 'DEW_POINT_10d', 
                     'DEW_POINT_4h', 'DEW_POINT_8h', 'DEW_POINT_16h'
                     ]]


print("Dataframe for Classification")
print(df_class.head(25))

#####
#START CLASSIFICATION
#####

sheet_id = '1IrZgmVjHmFkdxxM_65XzKW_nt8p8LCIHgkqtbjY4QJE'
sheet_name = 'trail_adjustments'
url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
df_gsheet_trail_adjustments = pd.read_csv(url)
trail_adjustments = dict(zip(df_gsheet_trail_adjustments['Trail'], df_gsheet_trail_adjustments['PRCP_ADJ_VALUE']))

def adjust_prcp(prcp, tmax, dew_point, trail):
    # Adjust based on TMAX
    if tmax > 85:
        prcp *= 0.8  
    elif tmax > 75:
        prcp *= 0.9  
    elif tmax <= 55:
        prcp *= 1.2  
    elif tmax <= 65:
        prcp *= 1.1 

    dew_point_temp_diff = tmax - dew_point
    # Adjust based on DEW_POINT
    if dew_point_temp_diff < 5:
        prcp *= 1.2
    elif dew_point_temp_diff < 10:
        prcp *= 1.1  
    elif dew_point_temp_diff < 15:
        prcp *= 1.0  
    elif dew_point_temp_diff < 20:
        prcp *= 0.9
    elif dew_point_temp_diff < 25:
        prcp *= 0.8  



    # trail_adjustments = dict(zip(df_gsheet_trail_adjustments['Trail'], df_gsheet_trail_adjustments['PRCP_ADJ_VALUE']))

    # Print the dictionary to verify
    # print(trail_adjustments)

    # # # Adjust based on trail
    # trail_adjustments = {
    #     'Devou Park': 1.1,
    #     'East Fork State Park': 1.3,
    #     'England Idlewild': 1.0,
    #     'Harbin Park': 1.0,
    #     'Landen Deerfield': 1.0,
    #     'Milford Trails': 0.4,
    #     'Mitchell Memorial Forest': 1.0,
    #     'Mount Airy Forest': 0.5,
    #     'Premier Health Bike Park': 1.2,
    #     'Tower Park': 1.0,
    #     'Caesar Creek': 1.0,
    # }
    
    # Apply trail adjustment if the trail is in the adjustment list
    if trail in trail_adjustments:
        prcp *= trail_adjustments[trail]

    return prcp


sheet_id = '1IrZgmVjHmFkdxxM_65XzKW_nt8p8LCIHgkqtbjY4QJE'
sheet_name = 'if_statements'
url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
df_gsheet_if_statements = pd.read_csv(url)
print(df_gsheet_if_statements.head(10))

def trail_status(row):
    prcp_4h = row['PRCP_4h']
    prcp_8h = row['PRCP_8h']
    prcp_16h = row['PRCP_16h']
    prcp_1d = row['PRCP_1d']
    prcp_2d = row['PRCP_2d']
    prcp_3d = row['PRCP_3d']
    prcp_5d = row['PRCP_5d']
    prcp_7d = row['PRCP_7d']
    tmax_4h = row['TMAX_4h']
    tmax_8h = row['TMAX_8h']
    tmax_16h = row['TMAX_16h']
    tmax_1d = row['TMAX_1d']
    tmax_2d = row['TMAX_2d']
    tmax_3d = row['TMAX_3d']
    tmax_5d = row['TMAX_5d']
    tmax_7d = row['TMAX_7d']
    dew_point_4h = row['DEW_POINT_4h']
    dew_point_8h = row['DEW_POINT_8h']
    dew_point_16h = row['DEW_POINT_16h']
    dew_point_1d = row['DEW_POINT_1d']
    dew_point_2d = row['DEW_POINT_2d']
    dew_point_3d = row['DEW_POINT_3d']
    dew_point_5d = row['DEW_POINT_5d']
    dew_point_7d = row['DEW_POINT_7d']
    trail = row['trail']

    # Adjust precipitation values
    prcp_4h = adjust_prcp(prcp_4h, tmax_4h, dew_point_4h, trail)
    prcp_8h = adjust_prcp(prcp_8h, tmax_8h, dew_point_8h, trail)
    prcp_16h = adjust_prcp(prcp_16h, tmax_16h, dew_point_16h, trail)
    prcp_1d = adjust_prcp(prcp_1d, tmax_1d, dew_point_1d, trail)
    prcp_2d = adjust_prcp(prcp_2d, tmax_2d, dew_point_2d, trail)
    prcp_3d = adjust_prcp(prcp_3d, tmax_3d, dew_point_3d, trail)
    prcp_5d = adjust_prcp(prcp_5d, tmax_5d, dew_point_5d, trail)
    prcp_7d = adjust_prcp(prcp_7d, tmax_7d, dew_point_7d, trail)

    # sheet_id = '1IrZgmVjHmFkdxxM_65XzKW_nt8p8LCIHgkqtbjY4QJE'
    # sheet_name = 'if_statements'
    # url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    # df_gsheet_if_statements = pd.read_csv(url)

    # print("gsheets if statements", df_gsheet_if_statements.head(10))

    # status_levels = ['DEFINITE CLOSE', 'LIKELY CLOSE', 'LIKELY WET/OPEN', 'LIKELY OPEN', 'DEFINITE OPEN']

    dimensions = ['prcp_4h', 'prcp_8h', 'prcp_16h', 'prcp_1d', 'prcp_2d', 'prcp_3d', 'prcp_5d', 'prcp_7d']
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
        for status in status_levels:
            if ((prcp_values['prcp_4h'] > greater_than['prcp_4h'][status] or prcp_values['prcp_4h'] < less_than['prcp_4h'][status]) and 
                (prcp_values['prcp_8h'] > greater_than['prcp_8h'][status] or prcp_values['prcp_8h'] < less_than['prcp_8h'][status]) and 
                (prcp_values['prcp_16h'] > greater_than['prcp_16h'][status] or prcp_values['prcp_16h'] < less_than['prcp_16h'][status]) and 
                (prcp_values['prcp_1d'] > greater_than['prcp_1d'][status] or prcp_values['prcp_1d'] < less_than['prcp_1d'][status]) and 
                (prcp_values['prcp_2d'] > greater_than['prcp_2d'][status] or prcp_values['prcp_2d'] < less_than['prcp_2d'][status]) and 
                (prcp_values['prcp_3d'] > greater_than['prcp_3d'][status] or prcp_values['prcp_3d'] < less_than['prcp_3d'][status])):
                return status
        return 'UNSURE - REVIEW IN PERSON'

    prcp_values = {
        'prcp_4h': prcp_4h,
        'prcp_8h': prcp_8h,
        'prcp_16h': prcp_16h,
        'prcp_1d': prcp_1d,
        'prcp_2d': prcp_2d,
        'prcp_3d': prcp_3d,
        'prcp_5d': prcp_5d,
        'prcp_7d': prcp_7d
    }
    print("processing this row of data", row)
    print("Result: ", get_trail_status(prcp_values))
    return get_trail_status(prcp_values)

# Your existing script processing
final_df['trail_status'] = final_df.apply(trail_status, axis=1)
print(final_df[['trail', 'PRCP_1d', 'PRCP_8h', 'PRCP_2d', 'PRCP_16h', 'PRCP_3d', 'PRCP_5d', 'PRCP_7d', 'TMAX_1d', 'DEW_POINT_1d', 'trail_status']])


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
        return pd.DataFrame(columns=['trail', 'PRCP_1d', 'PRCP_8h', 'PRCP_2d', 'PRCP_16h', 'PRCP_3d', 'PRCP_5d', 'PRCP_7d', 'TMAX_1d', 'DEW_POINT_1d', 'trail_status', 'timestamp'])

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

# Your existing script processing
final_df['trail_status'] = final_df.apply(trail_status, axis=1)
print(final_df[['trail', 'PRCP_1d', 'PRCP_8h', 'PRCP_2d', 'PRCP_16h', 'PRCP_3d', 'PRCP_5d', 'PRCP_7d', 'TMAX_1d', 'DEW_POINT_1d', 'trail_status']])

# Append the final_df to log
log_df = append_to_log(final_df[['trail', 'PRCP_1d', 'PRCP_8h', 'PRCP_2d', 'PRCP_16h', 'PRCP_3d', 'PRCP_5d', 'PRCP_7d', 'TMAX_1d', 'DEW_POINT_1d', 'trail_status']])

#### VIEW LOG

#### VIEW LOG
print("VIEW LOG ######")
# Get current timestamp and past 24 hours timestamp
current_timestamp = datetime.now().replace(minute=0, second=0, microsecond=0)
past_24_hours = current_timestamp - timedelta(hours=24)

# Filter log_df based on timestamp in past 24 hours and sort it
# the script runs once per hour, so duplicates should only exist when in DEV mode locally running it more than 1X per hour
# will drop duplicates at random to deal with this
log_df_for_email = log_df[log_df['timestamp'] >= past_24_hours.strftime('%Y-%m-%d %H:%M:%S')].sort_values(['trail', 'timestamp'], ascending=[True, False])[['trail', 'timestamp', 'trail_status']].drop_duplicates(subset=['trail', 'timestamp'])

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


######
# EMAIL
#######

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


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
run_openai_api = False  # Change this to False if you want to skip the OpenAI API call

if run_openai_api:
    openai.api_key = openai_api_key

    # Call the OpenAI API to generate summary
    df_summary_input = ("""Please point out trail status: {}""".format(log_df_for_email.to_string(index=False)))

    response = openai.ChatCompletion.create(
        model="gpt-4o",  
        temperature=0.4,
        max_tokens=1000,
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
        return f"<span style='color: darkred; font-weight: bold;'>{status}</span>"
    elif status == "LIKELY WET/OPEN":
        return f"<span style='color: goldenrod; font-weight: bold;'>{status}</span>"
    elif status == "LIKELY OPEN":
        return f"<span style='color: darkgreen; font-weight: bold;'>{status}</span>"
    elif status == "DEFINITE OPEN":
        return f"<span style='color: darkgreen; font-weight: bold;'>{status}</span>"
    else:
        return f"<span style='color: black;'>{status}</span>"

# Create the HTML content for changes
def get_trail_changes(df, trail):
    changes = []
    for i in range(len(df) - 1):
        current_status = format_trail_status(status_reverse_mapping.get(df.iloc[i]['trail_status'], 'Unknown'))
        next_status = format_trail_status(status_reverse_mapping.get(df.iloc[i + 1]['trail_status'], 'Unknown'))
        timestamp = df.iloc[i + 1]['timestamp']
        changes.append(f"{current_status} → {next_status} ({timestamp})")
    return "<br>".join(changes)

# trail_changes = log_df_for_email.groupby('trail').apply(lambda df: get_trail_changes(df, df['trail'].iloc[0])).to_dict()
trail_changes = dict(sorted(log_df_for_email.groupby('trail').apply(lambda df: get_trail_changes(df, df['trail'].iloc[0])).to_dict().items()))

bullet_points = df_filtered.apply(lambda row: f"<li><strong>{row['trail']}:</strong> {format_trail_status(row['trail_status'])}</li>", axis=1).tolist()
bullet_points_html = "<ul>" + "".join(bullet_points) + "</ul>"

# Prepare the email body with individual headers for each 'CORA Trail'
today_date = datetime.now().strftime("%Y-%m-%d")
email_body = f"""
<h2>{today_date} Weather Report</h2>
<h3>Trail Status Data:</h3>
{bullet_points_html}
<hr>
</h2>Last Two Days - See Changes Over Time:</h2>
"""

for trail, changes in trail_changes.items():
    current_status = format_trail_status(status_reverse_mapping[log_df_for_email[log_df_for_email['trail'] == trail].iloc[0]['trail_status']])
    email_body += f"<h2>{trail_reverse_mapping[trail]}:</h2>"
    email_body += f"<h3>Current Status: {current_status}</h3>"
    email_body += f"<p>Changes:<br>{changes}</p>"

email_body += "<hr><p>Happy biking!</p>"

# Creating the email message
msg = MIMEMultipart('alternative')
msg['From'] = from_email
msg['To'] = ", ".join(email_addresses)
msg['Subject'] = f'{today_date} CORA Trail Status Update'

# Attach HTML content
part = MIMEText(email_body, 'html')
msg.attach(part)

with smtplib.SMTP('smtp.gmail.com', 587) as server:  # Use the correct SMTP server and port
    server.starttls()
    server.login(from_email, gmail_api_key)   
    server.sendmail(from_email, email_addresses, msg.as_string())
print("Email sent successfully!")
