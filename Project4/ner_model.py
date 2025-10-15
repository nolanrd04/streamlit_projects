import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, 'tripadvisor_hotel_reviews_preprocessed.csv'))
df.drop(columns=['Review'], inplace=True) # Only keep processed reviews
print(df.head(5))
