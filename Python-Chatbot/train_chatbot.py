#!/usr/bin/python

# Import Tools Libraries
import numpy as np
import random
import pickle
import json

# Import Deep Learning Libraries
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# Import Language Libraries
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load Data from saved file
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

# Preprocess the Loaded Data from json format

# words = all vocab
words = []
# classes = intent behind words
classes = []
# documents = combination between patterns and intents
documents = []
ignore_letters = ['!','?',',','.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize Words
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # Add to Corpus
        documents.append((word, intent['tag']))
        # Add to Class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)

# Lemmatize, create and save vocabulary to be used

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))
print(len(documents),'Documents')
print(len(classes),'Classes', classes)
print(len(words),'Unique Lemmatized Words', words)

# Save data as pickle Files
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


# Create training and output data storage
training = []
output_empty = [0]*len(classes)

# bag of words for each sentence - training
for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # This results in a '0' for each tag and '1' for the current tag - for each pattern
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])
print('Training Data Created')

# Neural network Model Layers
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile Model: SGD with Nesterov
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train Model
history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', history)
print("model is trained")