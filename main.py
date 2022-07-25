# Based on TechWithTim tutorials on ML/AI/DL Chatbot
# After every changes done to .json data file you have to retrain the model

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

# GUI with tkinter
from tkinter import *

class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Chatbot")
        self.root.geometry("209x180")
        self.root.resizable(width=FALSE, height=FALSE)
        self.firstUsage = True

        self.root.eval('tk::PlaceWindow . center') # Place window in the center of the screen

        self.logo = Label(self.root, text="Hello!", font=("Poppins", 25, "bold"))
        self.startText = Label(self.root, text="How can I help you?", font=("Poppins", 18))
        self.entryBox = Entry(self.root, font=("Calibri", 16))

        self.sendButton = Button(self.root, command=self.send) # Unvisible send button, sending with enter/return key
        self.entryBox.bind("<Return>", self.send)

        self.chatWindow = Text(self.root, cursor="arrow", font=("Poppins", 17), spacing1=5, wrap=WORD, bd=0, padx=10, height="8", width="50")
        self.chatWindow.config(state=DISABLED)
        self.scrollbar = Scrollbar(self.root, command=self.chatWindow.yview, width=4) # Binding scrollbar to Chat window
        self.chatWindow['yscrollcommand'] = self.scrollbar.set

        self.logo.place(x=67, y=25)
        self.startText.place(x=23,y=60)
        self.entryBox.place(x=18, y=105, height=34, width=174)

        self.entryBox.focus()

        self.root.mainloop()

    def send(self, _event=None):
        msg = self.entryBox.get()
        self.entryBox.delete("0", END)

        if msg != '':
            if self.firstUsage:
                self.firstUsage=False
                self.root.geometry("329x520")

                self.logo.destroy()
                self.startText.config(text="Ask me anything you want", font=("Poppins", 18, "bold"))

                self.startText.place(x=48, y=12)
                self.chatWindow.place(x=2,y=47, height=418, width=316)
                self.scrollbar.place(x=318,y=51, height=409)
                self.entryBox.place(x=50, y=470, height=34, width=234)

            self.chatWindow.config(state=NORMAL)
            self.chatWindow.insert(END, "→ " + msg + "\n")
            self.chatWindow.config(state=DISABLED)
            self.chatWindow.yview(END)

            self.chatbot_response(msg)

    def printChat(self, msg):
        self.chatWindow.config(state=NORMAL)
        self.chatWindow.insert(END, "" + msg + "\n")
        self.chatWindow.config(state=DISABLED)
        self.chatWindow.yview(END)

    def chatbot_response(self, msg):
        results = model.predict([bag_of_words(msg, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            self.printChat(random.choice(responses))
        else:
            self.printChat("Please try another question")


with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = [] # Includes words from patterns - example questions from user
    labels = [] # Includes tags

    docs_x = [] # Includes words from patterns - example questions from user
    docs_y = [] # Includes tags

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = [] # Includes training data
    output = [] # Includes outputs for training data

    out_empty = [0 for _ in range(len(labels))] # Preparing array of 0 for output which will have probability for each tag

    for x, doc in enumerate(docs_x): # Creating bag of words for every tag
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# We are testing model based on training data and output
tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

App()