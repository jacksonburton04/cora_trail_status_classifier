import os
import pickle
from datetime import datetime, timedelta
import pandas as pd
import requests
import math


def fetch_historical_weather_data(pickle_file, df_trail_locations, api_key):
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
            for i in range(1, 15):
                date_var = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                response = get_weather_data(lat, lon, date_var, api_key)
                try:
                    temp_max = response['temperature']['max']
                    precipitation = response['precipitation']['total'] * 0.0393701
                    print("date + PRCP", date_var, precipitation, trail)
                    dew_point = calculate_dew_point(response['temperature']['afternoon'], response['humidity']['afternoon'])
                    data.append([trail, date_var, temp_max, precipitation, dew_point])  # snow_flag
                except KeyError as e:
                    # Fetch a default error message from the response, if available, otherwise use a specific KeyError message
                    error_msg = response.get('message', f'KeyError for {e}')
                    print(f"Error for {trail} on {date_var}: {error_msg}")

            historical_data = pd.DataFrame(data, columns=['Trail', 'DATE', 'MAX TEMPERATURE', 'TOTAL PRECIPITATION', 'DEW_POINT'])  # 'SNOW_FLAG'])
            historical_one_week_all_trails = pd.concat([historical_one_week_all_trails, historical_data])

        # Save to pickle
        with open(pickle_file, 'wb') as f:
            pickle.dump(historical_one_week_all_trails, f)
        print("Did not have today's historical data, pulling API data and saving to pickle file.")

    return historical_one_week_all_trails

def update_hourly_weather_data(df_trail_locations, api_key):
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
            print("Pickle file is older than 50 minutes. Fetching new data.")
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
                print("hour + PRCP", hour, precip_inches, trail)
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

    return hourly_weather


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
                    
    def calculate_freeze_thaw_points(group):
        max_lookback = 12  # Maximum hours to look back
        
        # Create a new series to store cumulative points for each row
        cumulative_points = []
        
        # Convert to numpy array for easier slicing
        points_array = group['freeze_thaw_points'].to_numpy()
        
        # Iterate through each row in the group
        for i in range(len(group)):
            # Get points for current row and up to max_lookback previous rows
            points_window = points_array[max(0, i-max_lookback+1):i+1]
            
            # Calculate cumulative points until we hit a negative value
            total_points = 0
            for point in reversed(points_window):
                if point < 0:
                    break
                total_points += point
            
            cumulative_points.append(total_points)
        
        return pd.Series(cumulative_points, index=group.index)
    
    # Apply calculation to each trail group
    hourly_df['freeze_thaw_points_cumulative'] = hourly_df.groupby('trail', group_keys=False).apply(
        lambda x: calculate_freeze_thaw_points(x)
    )

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


