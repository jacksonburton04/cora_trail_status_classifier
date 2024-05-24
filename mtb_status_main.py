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

warnings.filterwarnings('ignore')

config_file_path = 'data/dev_config.json'
# config_file_path = '/root/mtb_forecast_automated/data/prod_config.json'

with open(config_file_path, 'r') as config_file:
    config = json.load(config_file)

data_directory = config['paths']['data_directory']
os.chdir(data_directory)

lookback_days_list = [2, 3, 4, 5, 6, 7, 11, 18]
future_weather_prcp_cutoff = 0.01 # if less than .01 inch rain forecasted, ignore the rain


def calculate_rolling_metrics(model_df, lookback_days_list):
    """
    Calculates rolling sums and averages for specified columns over a range of lookback days.

    Parameters:
    - model_df: DataFrame containing the data.
    - lookback_days_list: List of integers representing the lookback days for rolling calculations.

    Returns:
    - model_df: DataFrame with added columns for rolling calculations.
    """
    for i in lookback_days_list:
        for col in ['PRCP', 'TMAX', 'DEW_POINT']:  # Corrected column name here
            if col in model_df.columns:  # Check if column exists in DataFrame
                new_col_name = f'{col}_{i}d'
                if col == 'PRCP':  # Apply sum for 'PRCP'
                    model_df[new_col_name] = model_df[col].rolling(window=i, min_periods=1).sum()
                elif col == 'DEW_POINT':  # Apply mean for 'DEW_POINT'
                    model_df[new_col_name] = model_df[col].rolling(window=i, min_periods=1).mean()
                else:  # Apply max for other columns (in this context, 'TMAX')
                    model_df[new_col_name] = model_df[col].rolling(window=i, min_periods=1).max()
    model_df.fillna(0, inplace=True)
    return model_df

s3_client = boto3.client('s3')
s3 = boto3.client('s3')
bucket_name = 'mtb-trail-condition-predictions'


### APIS (ENCRYPTED) 

## OPEN WEATHER API
from cryptography.fernet import Fernet
with open("encryption_openweather_key.txt", "rb") as f:
    encryption_key = f.read()
cipher_suite = Fernet(encryption_key)
with open("encrypted_openweather_api_key.txt", "rb") as f:
    encrypted_api_key = f.read()
decrypted_api_key = cipher_suite.decrypt(encrypted_api_key)
api_key = decrypted_api_key.decode()

## GMAIL API
with open("encryption_gmail_key.txt", "rb") as f:
    encryption_key = f.read()
cipher_suite = Fernet(encryption_key)
with open("encrypted_gmail_api_key.txt", "rb") as f:
    encrypted_api_key = f.read()
decrypted_api_key = cipher_suite.decrypt(encrypted_api_key)
gmail_api_key = decrypted_api_key.decode()

## CHATGPT API
with open("encryption_openai_key.txt", "rb") as f:
    encryption_key = f.read()
cipher_suite = Fernet(encryption_key)
with open("encrypted_openai_api_key.txt", "rb") as f:
    encrypted_api_key = f.read()
decrypted_api_key = cipher_suite.decrypt(encrypted_api_key)
openai_api_key = decrypted_api_key.decode()

import json

with open('config_emails.json', 'r') as file:
    config = json.load(file)
    email_addresses = config['email_addresses']
    print(email_addresses)

from_email = email_addresses[0]
print(from_email)

current_hour = datetime.now().hour
if current_hour in [12]:
    import openai
    openai.api_key = openai_api_key
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    user_content_input = ("""Tell me a funny dad joke. Then, I want you to tell me if it rained today in Cincinnati, Ohio. Then, I want a random mountain biking tip.""")
    response = openai.ChatCompletion.create(
        model="gpt-4o", # "gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=2500,
        messages=[
            {"role": "system", "content": "You write my daily email newsletter for local Cincinnati CORA mountain bikers."},
            {"role": "user", "content": user_content_input}
        ]
    )
    
    daily_summary = response.choices[0].message.content
    print(daily_summary)
    
    ######

    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    # Replace with your details
    from_email = from_email
    to_emails = email_addresses
    app_password = gmail_api_key

    ####################

    # Today's Date for Header
    today_date = datetime.now().strftime("%Y-%m-%d")

    # Prepare the email body with individual headers for each 'CORA Trail'
    email_body = f"<h2>{today_date} Weather Report</h2><p>Daily Summary:</p><p>{daily_summary}</p><hr>"


    # Creating the email message
    msg = MIMEMultipart('alternative')
    msg['From'] = from_email
    msg['To'] = to_emails
    # msg['To'] = ", ".join(to_emails)  # Join the list of emails into a single string separated by commas
    msg['Subject'] = f'{today_date} CORA Trail Report'

    ####################

    # Attach HTML content
    part = MIMEText(email_body, 'html')
    msg.attach(part)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:  # Use the correct SMTP server and port
        server.starttls()
        server.login(from_email, app_password)
        server.sendmail(from_email, to_emails, msg.as_string())

    print("Email sent successfully!")
else:
    print("Email not sent")


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
    if datetime.now() - file_mod_time < timedelta(hours=3):
        print("Pickle file has been modified in the past hour. Loading existing data.")
        with open(pickle_file, 'rb') as f:
            hourly_weather = pickle.load(f)
    else:
        print("Pickle file is older than 1 hour. Fetching new data.")
        hourly_weather = pd.DataFrame()
else:
    hourly_weather = pd.DataFrame()

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
            # humidity = hour['humidity']
            temp = hour['temp']
            # Append data to the new_data list
            new_data.append([trail, hour_time.strftime('%Y-%m-%d %H:%M'), precip_inches, temp]) #snow_flag

    new_hourly_data = pd.DataFrame(new_data, columns=['Trail', 'HOUR', 'PRECIPITATION', 'TEMPERATURE']) 
    hourly_weather = pd.concat([new_hourly_data, hourly_weather]).drop_duplicates(['Trail','HOUR']).reset_index(drop=True)

# Filter for the past two days
hourly_weather = hourly_weather[(pd.to_datetime(hourly_weather['HOUR']) >= start_date) & (pd.to_datetime(hourly_weather['HOUR']) <= end_date)]
hourly_weather['HOUR'] = pd.to_datetime(hourly_weather['HOUR'])
# Save to pickle file
with open(pickle_file, 'wb') as f:
    pickle.dump(hourly_weather, f)
print("Hourly data updated and saved to pickle file.")

print(hourly_weather.head(25))


########
# END
#########

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


hourly_weather['DATE'] = pd.to_datetime(hourly_weather['HOUR']).dt.strftime('%d/%m/%Y')

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

### ROLLING METRICS CALCULATED

def calculate_rolling_metrics(model_df, lookback_days_list, lookback_hours_list):
    """
    Calculates rolling sums and averages for specified columns over a range of lookback days
    and lookback hours.

    Parameters:
    - model_df: DataFrame containing the data.
    - lookback_days_list: List of integers representing the lookback days for rolling calculations.
    - lookback_hours_list: List of integers representing the lookback hours for rolling calculations.

    Returns:
    - model_df: DataFrame with added columns for rolling calculations.
    """
    # Ensure the 'DATE' column is in datetime format and sort the DataFrame by 'DATE'
    model_df['DATE'] = pd.to_datetime(model_df['DATE'])
    model_df.sort_values(by='DATE', inplace=True)

    # Add rolling metrics for past X days
    for i in lookback_days_list:
        for col in ['PRCP', 'TMAX', 'DEW_POINT']:  # Columns to calculate rolling metrics for
            if col in model_df.columns:  # Check if column exists in DataFrame
                new_col_name = f'{col}_{i}d'
                if col == 'PRCP':  # Apply sum for 'PRCP'
                    model_df[new_col_name] = model_df[col].rolling(window=f'{i}D', on='DATE', min_periods=1).sum()
                elif col == 'DEW_POINT':  # Apply mean for 'DEW_POINT'
                    model_df[new_col_name] = model_df[col].rolling(window=f'{i}D', on='DATE', min_periods=1).mean()
                else:  # Apply max for other columns (in this context, 'TMAX')
                    model_df[new_col_name] = model_df[col].rolling(window=f'{i}D', on='DATE', min_periods=1).max()

    # Add rolling metrics for past X hours
    for i in lookback_hours_list:
        for col in ['PRCP', 'TMAX', 'DEW_POINT']:  # Columns to calculate rolling metrics for
            if col in model_df.columns:  # Check if column exists in DataFrame
                new_col_name = f'{col}_{i}h'
                if col == 'PRCP':  # Apply sum for 'PRCP'
                    model_df[new_col_name] = model_df[col].rolling(window=f'{i}H', on='DATE', min_periods=1).sum()
                elif col == 'DEW_POINT':  # Apply mean for 'DEW_POINT'
                    model_df[new_col_name] = model_df[col].rolling(window=f'{i}H', on='DATE', min_periods=1).mean()
                else:  # Apply max for other columns (in this context, 'TMAX')
                    model_df[new_col_name] = model_df[col].rolling(window=f'{i}H', on='DATE', min_periods=1).max()

    model_df.fillna(0, inplace=True)
    return model_df

lookback_days_list = [2, 4, 7]  # Example lookback days
lookback_hours_list = [6, 12, 18]  # Example lookback hours
weather_data_main = calculate_rolling_metrics(weather_data_main, lookback_days_list, lookback_hours_list)
weather_data_main.to_csv("weather_data_main.csv", index = False)
print(weather_data_main.head(30))

