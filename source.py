# %%
import numpy as np
import pandas as pd

# %%
import re
import nltk
import pandas as pd
from textblob import Word
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras import layers
from sklearn.model_selection import train_test_split 
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
data = pd.read_csv('data/financial_phrasebank.csv')
data = data[data['label'].isin([1,2])]

# %%
def text_cleaning(df):
    df['sentence'] = df['sentence'].apply(lambda x: x.lower())
    df['sentence'] = df['sentence'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
    df['sentence'] = df['sentence'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
    df['sentence'] = df['sentence'].str.replace('\n', '')
    df['sentence'] = df['sentence'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))
    return df

stop_words = stopwords.words('english')
cleaned = text_cleaning(data)
#cleaned['sentence'][:5]

# %%
max_len = 120

tokenizer = Tokenizer(num_words=500, split=' ') 
tokenizer.fit_on_texts(cleaned['sentence'].values)

X = tokenizer.texts_to_sequences(cleaned['sentence'].values)
X = pad_sequences(X, maxlen=max_len)

# %%
Y = pd.get_dummies(cleaned['label']).values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#print(X_train.shape,Y_train.shape)
#print(X_test.shape,Y_test.shape)

# %%
vocab_size = 2000

model = Sequential()
model.add(layers.Embedding(max_features, emb_dim, input_length=X.shape[1]))
model.add(layers.SpatialDropout1D(0.7))
model.add(layers.LSTM(64, dropout=0.7, recurrent_dropout=0.7))
model.add(layers.Dense(2, activation='sigmoid'))

# %%
model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['binary_accuracy'])

#print(model.summary())

# %%
epochs = 100
batch_size = 256

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=10, # how many epochs to wait before stopping
    restore_best_weights=True,
)

history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_split=0.2,
                    callbacks=[early_stopping],
)

# %%
#score = model.evaluate(X_test, Y_test, verbose=1)

# %%
history_df = pd.DataFrame(history.history)
#history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
#history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")

# %%
twt = input('Entrez une phrase : ')
twt = tokenizer.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=max_len)

sentiment = model.predict(twt, batch_size=1, verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("Négatif")
elif (np.argmax(sentiment) == 1):
    print("Positif")