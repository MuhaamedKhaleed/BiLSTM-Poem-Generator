import numpy as np
import tensorflow.keras.utils as ku
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


# Reading the text data file
data = open('poem.txt', encoding="utf8").read()

# Generating the corpus by splitting the text into lines
corpus = data.lower().split("\n")

# Remove duplicates
corpus = list(set(corpus))


# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print("Total Words:", total_words)

# Create input sequences using list of tokens
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and label
predictors = input_sequences[:, :-1]  # All but the last word are predictors
label = input_sequences[:, -1]  # Only the last word is the label

# One-hot encode the labels
label = ku.to_categorical(label, num_classes=total_words)

# Build the Bi-Directional LSTM Model using Input layer
input_layer = Input(shape=(max_sequence_len-1,))

embedding_layer = Embedding(input_dim=total_words, output_dim=100)(input_layer)
bi_lstm = Bidirectional(LSTM(150, return_sequences=True))(embedding_layer)
dropout_layer = Dropout(0.2)(bi_lstm)
lstm_layer = LSTM(100)(dropout_layer)
dense_layer = Dense(total_words//2, activation='relu', kernel_regularizer=regularizers.l2(0.01))(lstm_layer)
output_layer = Dense(total_words, activation='softmax')(dense_layer)

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train the model
history = model.fit(predictors, label, epochs=150, verbose=1)

# Evaluation accuracy
loss, accuracy = model.evaluate(predictors, label, verbose=0)
print('Accuracy: %f' % (accuracy * 100))

# Text generation
seed_text = "his drumming"
next_words = 8

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)

    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break

    seed_text += " " + output_word

print(seed_text)

