import pandas as pd

# Load dataset

# Combine 25 headlines into one

headlines.append(' '.join(str(x) for x in df.iloc[row, 2:27]))

from sklearn.feature_extraction.text import CountVectorizer

X = vectorizer.fit_transform(headlines).toarray()

y = df['Label'].values

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

# Build model

model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

import matplotlib.pyplot as plt

# Accuracy plot

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

# Loss plot

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



 Advanced Model - LSTM for Sequential Text

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, LSTM, Dense

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenization

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(headlines)

X = tokenizer.texts_to_sequences(headlines)

X = pad_sequences(X, maxle…

# Train-test split

# LSTM Model

model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=500),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

 Extra - EarlyStopping and ModelCheckpoint

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=3)

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test),
                    callbacks=[early_stop, checkpoint])

 Extra - Confusion Matrix and Classification Report

from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns

# Predictions

# Confusion Matrix

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()

# Classification Report

from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

from sklearn.feature_extraction.text import TfidfVectorizer

X = tfidf.fit_transform(headlines).toarrar

model.save('stock_prediction_model.h5')

from tensorflow.keras.models import load_model

ID: STU659638739d8f91704343667