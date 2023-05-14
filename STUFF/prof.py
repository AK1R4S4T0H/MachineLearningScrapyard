
def Professor(dataset_path, num_epochs=15, batch_size=32):
  from keras.backend import categorical_crossentropy
  import tensorflow as tf
  import pandas as pd
  import numpy as np
  import io
  import pickle
  import requests
  from keras.models import Sequential
  from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense

  df = pd.read_csv(dataset_path, header=None, names=["text", "label"], delimiter='\t')
  print(df["label"].unique())
  # Tokenize
  tokenizer = tf.keras.preprocessing.text.Tokenizer()
  tokenizer.fit_on_texts(df["text"])
  sequences = tokenizer.texts_to_sequences(df["text"])
  word_index = tokenizer.word_index

  max_length = max([len(seq) for seq in sequences])
  X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length)


  # Convert to one-hot encoding
  vocab_size = len(tokenizer.word_index) + 1
  y = df["label"].values

  vocab_size = len(tokenizer.word_index) + 1
  embedding_dim = 100
  max_length = X.shape[1]
  model = Sequential()
  model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
  model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(1, activation='softmax'))

  # Compile
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

  # Train
  model.fit(X, y, epochs=num_epochs, batch_size=batch_size)

  # Save
  model.save('Professor.h5')

  # Save tokenizer
  with open('tokenP.pickle', 'wb') as handle:
      pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

  print("Number of tokens generated:", len(tokenizer.word_index))
  
  return model, tokenizer
Professor('/content/content/AllBooks_baseline_DTM_Labelled.csv')