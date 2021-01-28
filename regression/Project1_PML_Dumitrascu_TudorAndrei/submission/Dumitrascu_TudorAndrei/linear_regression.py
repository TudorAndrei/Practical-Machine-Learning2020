import re

import emoji
import nltk
import numpy as np
import pandas as pd
import spacy
import tqdm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

# Download the stopwords
nltk.download('stopwords')

# Load the data
path_to_data = '/content/drive/MyDrive/pml/data/base/'
train_data = pd.read_csv(f'{path_to_data}/train.txt', header=None)
train_data.columns = ["id", "lat", "long", "text"]
validation_data = pd.read_csv(f"{path_to_data}/val.txt", header=None)
validation_data.columns = ["id", "lat", "long", "text"]
test_data = pd.read_csv(f"{path_to_data}/test.txt", header=None)
test_data.columns = ["id", "text"]

stop_words = stopwords.words('german')


def to_lower(text):
    return text.lower()


def remove_extra_whitespaces(text):
    return ' '.join(text.strip().split())


def remove_emojis(text):
    text = text.encode('utf8')
    return emoji.get_emoji_regexp().sub(r'', text.decode('utf8'))


def remove_stopwords(text, stopwords=stop_words):
    return ' '.join([w for w in text.split() if not w in stopwords])


def remove_punctuation(text):
    return re.sub('[^\w\s]', "", text)


def remove_word_shorter_than(text):
    aux = [i for i in text.split(" ") if len(i) > 2]
    return " ".join(aux)


def stemming(text):
    stemmer = nltk.stem.cistem.Cistem()
    words = []
    for word in text.split(" "):
        words.append(stemmer.stem(word))
    text = " ".join(words)
    return text


def replace_hyphens(text):
    return text.replace('-', " ")


def remove_numbers(text):
    return''.join([i for i in text if not i.isdigit()])


def encoding(text, encoder, vocab):
    valid_words = []
    for word in text.split(" "):
        if word in vocab:
            valid_words.append(word)
    encoded = encoder.transform([" ".join(valid_words)]).toarray()
    return encoded[0].tolist()


def preprocess(text, methods=None, encoder=None):
    functions = {
        # 'lower': to_lower,
        'whitespaces': remove_extra_whitespaces,
        'emojis': remove_emojis,
        'number': remove_numbers,
        'hyphens': replace_hyphens,
        'chars': remove_word_shorter_than,
        'punctuation': remove_punctuation,
        'stopwords': remove_stopwords,
        'stemming': stemming,

    }

    if methods is None:
        methods = list(functions.keys())

    for method in methods:
        text = functions[method](text)
    return text


def process_dfs(train_df, val_df, test_df):
    # location = '/content/drive/MyDrive/pml/data/labeled/'
    print("~~~~~Starting")

    train_df.iloc[:, 3] = train_df['text'].apply(lambda x: preprocess(x))

    words = train_df.iloc[:, 3]
    encoder = HashingVectorizer(norm='l1', n_features=2**10).fit(words)
    vocab = set(" ".join(words).split(" "))

    train_df.iloc[:, 3] = train_df.iloc[:, 3].apply(
        lambda x: encoding(x, encoder, vocab)).to_numpy()
    # train_df = pd.concat([train_df.drop(columns='text'), pd.DataFrame(train_df['text'].tolist(), index=train_df.index).add_prefix('x')], axis=1)

    print("~~~~~Finished train set")
    val_df.iloc[:, 3] = val_df['text'].apply(lambda x: preprocess(x))
    val_df.iloc[:, 3] = val_df.iloc[:, 3].apply(
        lambda x: encoding(x, encoder, vocab)).to_numpy()
    # val_df = pd.concat([val_df.drop(columns='text'), pd.DataFrame(val_df['text'].tolist(), index=val_df.index).add_prefix('x')], axis=1)

    print("~~~~~Finished val set")
    test_df.iloc[:, 1] = test_df['text'].apply(lambda x: preprocess(x))
    test_df.iloc[:, 1] = test_df.iloc[:, 1].apply(
        lambda x: encoding(x, encoder, vocab)).to_numpy()
    # test_df = pd.concat([test_df.drop(columns='text'), pd.DataFrame(test_df['text'].tolist(), index=test_df.index).add_prefix('x')], axis=1)

    print("~~~~~Saving")
    # Save the models
    # train_df.to_csv(f"{location}/train.csv", index=None)
    # val_df.to_csv(f"{location}/val.csv", index=None)
    # test_df.to_csv(f"{location}/test.csv", index=None)
    # print("~~~~~Done")
    return train_df, val_df, test_df


train, val, test = process_dfs(train_data, validation_data, test_data)


X_train = train.text.tolist()
X_val = val.text.tolist()
X_test = test.text.tolist()

target_cols = ['lat', 'long']
y_train = train[target_cols]
y_val = val[target_cols]


pipe = Pipeline([('scaler', MinMaxScaler())])
pipe.fit(X_train)

X_train = pipe.transform(X_train)
X_val = pipe.transform(X_val)
X_test = pipe.transform(X_test)

lr = LinearRegression(n_jobs=-1)
lr.fit(X_train, y_train)

y_hat = lr.predict(X_val)
print(mean_absolute_error(y_hat, y_val))
print(mean_squared_error(y_hat, y_val))

predicted = lr.predict(X_test)

# Create a submission file

result_df = test_data.copy(deep=True)
result_df['lat'] = 0.
result_df['long'] = 0.
nr = 1


for i in tqdm(range(len(result_df.index))):

    result_df.iat[i, 2] = round(float(predicted[i][0]), 15)
    result_df.iat[i, 3] = round(float(predicted[i][1]), 15)

result_df.drop(['text'], axis=1).to_csv(f'submission{nr}.txt', index=False)
