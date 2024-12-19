import os
import pandas as pd
from tqdm import tqdm

INPUTS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'inputs')

DS_NAME = 'news'
NEWS_VENUE = 'yahoo-finance'

target_path = os.path.join(INPUTS_PATH, DS_NAME, NEWS_VENUE)
years = os.listdir(target_path)

# get all the years available - assume the user has downloaded the raw jsons and parsed them using "load_and_break_news_data.py"
years_data = []
for year in tqdm(years, total=len(years), desc=f'Loading Sentiment Data for {NEWS_VENUE} Venue...'):
    files = os.listdir(os.path.join(target_path, year))

    moths_data = []
    for file in files:
        data = pd.read_csv(os.path.join(target_path, year, file))

        # select necessary columns for the analysis
        selected_cols = ['date_publish', 'title', 'description', 'maintext',
                        'mentioned_companies', 'named_entities', 'sentiment', 'emotion']
        selected_data = data[selected_cols]

        # go over all the rows of the news dataset and select the needed columns for the analysis
        tmp_final_data = []
        for idx, row in selected_data.iterrows():
            companies = row['mentioned_companies']
            companies = companies.replace('[', '').replace(']', '').replace("'", "").replace(' ', '').split(',')
            tmp_data = []
            for company in companies:
                tmp_data.append(pd.DataFrame(
                    [
                        {
                            'date_publish': row['date_publish'],
                            'company': company,
                            'dummy': 1,
                            'negative_sentiment': float(row['sentiment'].split(', ')[0].split(": ")[-1].replace('}', '')),
                            'neutral_sentiment': float(row['sentiment'].split(', ')[1].split(": ")[-1].replace('}', '')),
                            'positive_sentiment': float(row['sentiment'].split(', ')[2].split(": ")[-1].replace('}', '')),
                            'negative_emotion': float(row['emotion'].split(', ')[0].split(": ")[-1].replace('}', '')),
                            'neutral_emotion': float(row['emotion'].split(', ')[1].split(": ")[-1].replace('}', '')),
                            'positive_emotion': float(row['emotion'].split(', ')[2].split(": ")[-1].replace('}', '')) 
                        }
                    ]
                ))
            tmp_data = pd.concat(tmp_data, axis=0)
            tmp_final_data.append(tmp_data)
        tmp_final_data = pd.concat(tmp_final_data, axis=0)
        years_data.append(tmp_final_data)
sentiment_data = pd.concat(years_data, axis=0)
sentiment_data['venue'] = NEWS_VENUE

cols_order = [
       'date_publish', 'venue', 'company', 'dummy', 'negative_sentiment',
       'neutral_sentiment', 'positive_sentiment', 'negative_emotion',
       'neutral_emotion', 'positive_emotion'
]

output_path = os.path.join(os.path.dirname(__file__), 'data', 'inputs', DS_NAME, NEWS_VENUE, 'processed')
os.makedirs(output_path, exist_ok=True)

sentiment_data[cols_order].to_csv(os.path.join(output_path, f'{NEWS_VENUE}_sentiment.csv'), index=False)