
from google.colab import drive
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from knockknock import telegram_sender
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from tqdm import tqdm, trange
from transformers import AutoTokenizer, TFAutoModel
from xgboost import XGBRegressor

SEQUENCE_LENGTH = 128


# Read the files
type = "cased"
ftyp = "csv"
path = f'/gdrive/MyDrive/pml/data/{type}'
train_ds = pd.read_csv(f"{path}/train.{ftyp}")
val_ds = pd.read_csv(f"{path}/val.{ftyp}")
test_ds = pd.read_csv(f"{path}/test.{ftyp}")
train_ds.columns = ["id", "lat", "long", "text"]
val_ds.columns = ["id", "lat", "long", "text"]
test_ds.columns = ["id", "text"]
y_train = train_ds.drop(["id", "text"], axis=1)
y_val = val_ds.drop(["id", "text"], axis=1)

max = 0
for i in range(len(train_ds.index)):
    wordlen = len(train_ds.iat[i, 3].split(" "))
    if wordlen > max:
        max = wordlen

print(max)

# Load the pretrained model and tokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
bert = TFAutoModel.from_pretrained("dbmdz/bert-base-german-cased")


def tokenize(sentence):
    # generate the tokens and the attentions mask
    tokens = tokenizer.encode_plus(sentence, max_length=SEQUENCE_LENGTH,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')
    return tokens['input_ids'], tokens['attention_mask']


def generate_id_mask(dataframe):

    X_id = np.zeros((len(dataframe), SEQUENCE_LENGTH))
    X_att_mask = np.zeros((len(dataframe), SEQUENCE_LENGTH))

    # loop through data and tokenize everything
    for i, sentence in tqdm(enumerate(dataframe['text']), total=len(dataframe)):
        X_id[i, :], X_att_mask[i, :] = tokenize(sentence)

    return X_id, X_att_mask


def create_submission(attempt, model):
    # generate a file that can be submitted to the kaggle platform
    result_df = test_ds.copy(deep=True)
    result_df['lat'] = 0.
    result_df['long'] = 0.

    for i in tqdm(range(len(result_df.index))):
        # extract the text
        text_to_predict = result_df.text[i]
        xis, xmask = tokenize(text_to_predict)
        result = model.predict([xis, xmask])
        result_df.iat[i, 2] = round(float(result[0][0]), 15)
        result_df.iat[i, 3] = round(float(result[0][1]), 15)

    result_df.drop(['text'], axis=1).to_csv(
        f"result{attempt}.txt", index=False)


# @telegram_sender(token=TOKEN, chat_id=CHAT_ID)
def finetune(model, epochs=20, lr=.001, error="mae"):
    # Finetune the model

    # Unfreeze the bert layer
    model.layers[2].trainable = True

    loss = tf.keras.losses.MeanSquaredError()

    if loss == "msle":
        loss = tf.keras.losses.MeanSquaredLogarithmicError()

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae", "mse"])

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_mae', patience=3, restore_best_weights=True)

    history = model.fit([X_ids_train, X_attn_mask_train], y_train,
                        epochs=epochs,
                        callbacks=[es],
                        validation_data=(
                            [X_ids_test, X_attn_mask_test], y_val),
                        batch_size=64)

    training_loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]
    return [training_loss, validation_loss], model, history


X_ids_train, X_attn_mask_train = generate_id_mask(train_ds)
X_ids_test, X_attn_mask_test = generate_id_mask(val_ds)


# @telegram_sender(token=TOKEN, chat_id=CHAT_ID)
def start_training(epochs=20):
    # Generate the model and train it

    input_ids = tf.keras.layers.Input(
        shape=(SEQUENCE_LENGTH,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(
        shape=(SEQUENCE_LENGTH,), name='attention_mask', dtype='int32')

    # we only keep tensor 0 (last_hidden_state)
    bert_layer = bert.bert(input_ids, attention_mask=mask)[0]

    X = tf.keras.layers.LSTM(512)(bert_layer)
    y = tf.keras.layers.Dense(2)(X)

    model = tf.keras.Model(inputs=[input_ids, mask], outputs=[y])

    model.layers[2].trainable = False
    loss = tf.keras.losses.MeanSquaredLogarithmicError()

    model.compile(optimizer="adam", loss=loss, metrics=["mse", 'mae'])

    history = model.fit([X_ids_train, X_attn_mask_train], y_train,
                        epochs=epochs,
                        validation_data=(
                            [X_ids_test, X_attn_mask_test], y_val),
                        batch_size=128)

    training_loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]
    return [training_loss, validation_loss], model, history


# Train the model
loss, lstm_model, history = start_training(10)
# Finetune the model
loss, lstm_mode, history = finetune(lstm_mode, error='msle')


# plot the history
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
epochs = range(1, 10 + 0)

training_loss = history.history['loss'][1:]
val_loss = history.history['val_loss'][1:]

training_mae = history.history['mae'][1:]
val_mae = history.history['val_mae'][1:]

training_mse = history.history['mse'][1:]
val_mse = history.history['val_mse'][1:]

ax1.set_title('MSLE Loss')
ax1.plot(epochs, training_loss, 'r-', label='train')
ax1.plot(epochs, val_loss, 'b-', label='val')
ax1.legend()


ax2.set_title('MAE')
ax2.plot(epochs, training_mae, 'r-', label='train')
ax2.plot(epochs, val_mae, 'b-', label='val')
ax2.legend()

ax3.set_title('MSE')
ax3.plot(epochs, training_mse, 'r-', label='train')
ax3.plot(epochs, val_mse, 'b-', label='val')
ax3.legend()


plt.xlabel('Epoch')
plt.show()
