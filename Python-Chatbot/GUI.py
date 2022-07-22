# Load NLP libraries and Tools
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random

# Load Model and saved data
from tensorflow.keras.models import load_model
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# Initialise lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize Function
def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Bag of Words Function
def bag_words(sentence, words, show_details=True):
    sentence_words = clean_sentence(sentence)
    bag = [0]*len(words)
    for each_word in sentence_words:
        for i,word in enumerate(words):
            if word == each_word:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %each_word" % word)
    return (np.array(bag))

# Prediction function and thresholding
def predict_classes(sentence):
    bag = bag_words(sentence, words, show_details=False)
    results = model.predict(np.array([bag]))[0]
    threshold = 0.25
    # Retrieve predictions above threshold value and sort them
    results = [[i,r] for i,r in enumerate(results) if r > threshold]
    results.sort(key=lambda x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent":classes[r[0]], "probability":str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

# Create GUI using Tkinter library
import tkinter
from tkinter import *

def send():
    message = EntryBox.get("1.0", "end-1c").strip()
    EntryBox.delete("0.0", END)
    
    if message != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + message +'\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12))
        
        ints = predict_classes(message)
        response = getResponse(ints, intents)
        
        ChatBox.insert(END, "Chatbot: " + response + '\n\n')
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)

# Initialize and set parameters for GUI bot
root = Tk()
root.title("Chatbot")
root.geometry("600x500")
root.resizable(width=FALSE, height=FALSE)

# Create Window
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial")
ChatBox.config(state=DISABLED)

# Add Scrollbar
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

# Create Send Button
SendButton = Button(root, font=("Verdana",12,'bold'), text="Send", width="12", height=5, bd=0, bg="#f9a602", activebackground="#3c9d9b",fg='#000000',command=send)

# Create Entry Box for message
EntryBox = Text(root, bd=0, bg="white", width="45", height="5", font="Arial")
EntryBox.bind("<Return>",send)

# Build Screen Components
scrollbar.place(x=584, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=584)
EntryBox.place(x=128, y=401, height=90, width=584)
SendButton.place(x=6, y=401, height=90)
root.mainloop()