import os
import pandas as pd

INPUTS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'inputs')

DS_NAME = 'news'
NEWS_VENUE = 'yahoo-finance'

target_path = os.path.join(INPUTS_PATH, DS_NAME, NEWS_VENUE)
files = os.listdir(target_path)

for file in files:
    year = file.split('_')[0]
    data = pd.read_json(os.path.join(target_path, file))

    output_path = os.path.join(os.path.dirname(__file__), 'data', 'inputs', DS_NAME, NEWS_VENUE, year)
    os.makedirs(output_path, exist_ok=True)

    data['date_publish'] = pd.to_datetime(data['date_publish'])
    data['month_publish'] = data['date_publish'].dt.month

    for month in data['month_publish'].unique():
        month_data = data[data['month_publish'] == month]
        month_data.to_csv(os.path.join(output_path, f'{month}.csv'), index=False)