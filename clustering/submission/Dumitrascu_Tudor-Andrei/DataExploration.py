from collections import Counter

import numpy as np
import pandas as pd

path = 'reviews'
# artists = pd.read_csv(f'{path}/artists.csv')
content = pd.read_csv(f'{path}/content.csv')
genres = pd.read_csv(f'{path}/genres.csv')
# labels = pd.read_csv(f'{path}/labels.csv')
# reviews = pd.read_csv(f'{path}/reviews.csv')
# years = pd.read_csv(f'{path}/years.csv')


genres.drop_duplicates(["reviewid"], inplace=True, ignore_index=True)
content.drop_duplicates(["content"], inplace=True, ignore_index=True)


dataset = pd.merge(content, genres, on='reviewid', how='outer')
content.drop_duplicates(["reviewid"], inplace=True, ignore_index=True)
# dataset = pd.merge(dataset, labels, on='reviewid', how='outer')
# dataset = pd.merge(dataset, reviews, on='reviewid', how='outer')
# dataset = pd.merge(dataset, years, on='reviewid', how='outer')
dataset.dropna(axis=0, how='any', inplace=True)


dataset.genre.value_counts().plot(kind='bar')
print(Counter(dataset.genre).most_common(10))


dataset[['content', 'genre']].to_csv(
    "data/content_dataset_full.csv", index=False)
