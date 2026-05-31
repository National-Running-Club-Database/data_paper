import pandas as pd
import os

def parse_time(time_str):
    if pd.isna(time_str):
        return float('nan')
    parts = str(time_str).split(':')
    try:
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h*3600 + m*60 + s
        elif len(parts) == 2:
            m, s = map(float, parts)
            return m*60 + s
        else:
            return float(time_str)
    except Exception:
        return float('nan')

def format_parsed_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:05.2f}"
    elif minutes > 0:
        return f"{minutes}:{seconds:05.2f}"
    else:
        return f"{seconds:05.2f}"

def get_course_details(row, course_details_df):
    match = course_details_df[(course_details_df['meet_id'] == row['meet_id']) &
                              (course_details_df['running_event_id'] == row['running_event_id']) &
                              (course_details_df['gender'] == row['gender'])]
    if not match.empty:
        return match.iloc[0].to_dict()
    match = course_details_df[(course_details_df['meet_id'] == row['meet_id']) &
                              (course_details_df['running_event_id'] == row['running_event_id'])]
    if not match.empty:
        return match.iloc[0].to_dict()
    return {}

def adjust_time_for_race(event_name:str, time:str, course_details:dict, gender:str):
    time = parse_time(time)
    event_dist = None
    # Handle NaN or None values
    if pd.isna(event_name) or event_name is None:
        return time  # Return original time if event_name is invalid
    # Convert to string if not already
    event_name = str(event_name)
    if event_name.endswith('m'):
        event_dist = float(event_name.replace('m', '').strip())
    elif event_name.endswith('mi'):
        event_dist = float(event_name.replace('mi', '').strip()) * 1609.34
    if pd.notna(course_details.get('temperature')) and pd.notna(course_details.get('dew_point')):
        weather_factor = course_details['temperature'] + course_details['dew_point']
        if weather_factor > 100:
            percent_increase = 0.0015 * (weather_factor - 100) ** 2
            time *= 1 - (percent_increase / 100)
    return time

def get_event_dist(event_name):
    # Handle NaN or None values
    if pd.isna(event_name) or event_name is None:
        return None
    # Convert to string if not already
    event_name = str(event_name)
    if event_name.endswith('m'):
        return float(event_name.replace('m', '').strip())
    elif event_name == '4 Mile':
        return 4 * 1609.34
    elif event_name == '5 Mile':
        return 5 * 1609.34
    else:
        return None

    if results_df is None or 'meet_id' not in results_df.columns:
        results_df = pd.read_csv(os.path.join('data', 'result.csv'))
    if meet_df is None:
        meet_df = pd.read_csv(os.path.join('data', 'meet.csv'))
    if athlete_df is None:
        athlete_df = pd.read_csv(os.path.join('data', 'athlete.csv'))
    if running_event_df is None:
        running_event_df = pd.read_csv(os.path.join('data', 'running_event.csv'))
    if 'meet_id' not in results_df.columns:
        raise ValueError("Input DataFrame to convert_exclude_nationals must have a 'meet_id' column.")
    non_nationals_meets = meet_df[~meet_df['nationals'].astype(bool)]['meet_id']
    filtered_results = results_df[results_df['meet_id'].isin(non_nationals_meets)].copy()
    filtered_results = filtered_results.merge(athlete_df[['athlete_id', 'gender']], on='athlete_id', how='left')
    filtered_results = filtered_results.merge(running_event_df[['running_event_id', 'event_name']], on='running_event_id', how='left')
    filtered_results = filtered_results.merge(meet_df[['meet_id', 'start_date']], on='meet_id', how='left')
    return standardize_and_convert_to_6k_8k(filtered_results, course_details_df=pd.DataFrame(), athlete_df=athlete_df, running_event_df=running_event_df, meet_df=meet_df)