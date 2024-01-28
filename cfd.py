from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

main = tkinter.Tk()
main.title("Credit Card Fraud Detection Using AdaBoost and Majority Voting")
main.geometry("1300x1200")

global filename
global cls
global X, Y, X_train, X_test, y_train, y_test
global ada_acc  # all global variables names define in above lines
global clean
global attack
global total

def traintest(train):  # method to generate test and train data from dataset
    X = train.values[:, 0:29]
    Y = train.values[:, 30]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    return X, Y, X_train, X_test, y_train, y_test

def generateModel():  # method to read dataset values which contains all five features data
    global X, Y, X_train, X_test, y_train, y_test
    train = pd.read_csv(filename)
    X, Y, X_train, X_test, y_train, y_test = traintest(train)
    text.insert(END, "Train & Test Model Generated\n\n")
    text.insert(END, "Total Dataset Size : " + str(len(train)) + "\n")
    text.insert(END, "Split Training Size : " + str(len(X_train)) + "\n")
    text.insert(END, "Split Test Size : " + str(len(X_test)) + "\n")

def upload():  # function to upload credit card dataset
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n");

def prediction(X_test, cls):  # prediction done here
    y_pred = cls.predict(X_test)
    for i in range(50):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred

# Function to calculate accuracy
def cal_accuracy(y_test, y_pred, details):
    accuracy = accuracy_score(y_test, y_pred) * 100
    text.insert(END, details + "\n\n")
    text.insert(END, "Accuracy : " + str(accuracy) + "\n\n")
    return accuracy

def runAdaBoost():
    global ada_acc
    global cls
    global X, Y, X_train, X_test, y_train, y_test
    cls = AdaBoostClassifier(n_estimators=50, random_state=0)
    cls.fit(X_train, y_train)
    text.insert(END, "Prediction Results (AdaBoost)\n\n")
    prediction_data = prediction(X_test, cls)
    ada_acc = cal_accuracy(y_test, prediction_data, 'AdaBoost Accuracy')

def runVoting():
    global voting_acc
    global cls
    global X, Y, X_train, X_test, y_train, y_test
    ada_clf = AdaBoostClassifier(n_estimators=50, random_state=0)
    majority_clf = RandomForestClassifier(n_estimators=50, random_state=0)  # You can use any classifier here
    clf = VotingClassifier(estimators=[('ada', ada_clf), ('majority', majority_clf)], voting='soft')
    clf.fit(X_train, y_train)
    text.insert(END, "Prediction Results (Majority Voting)\n\n")
    prediction_data = prediction(X_test, clf)
    voting_acc = cal_accuracy(y_test, prediction_data, 'Majority Voting Accuracy')
def predicts():
    global clean
    global attack
    global total
    clean = 0
    attack = 0
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(filename)
    test = test.values[:, 0:29]
    total = len(test)
    text.insert(END, filename + " test file loaded\n")
    y_pred = cls.predict(test)
    for i in range(len(test)):
        if str(y_pred[i]) == '1.0':
            attack += 1
            text.insert(END, "X=%s, Predicted = %s" % (test[i], 'Contains Fraud Transaction Signature So Transaction Stopped ') + "\n\n")
        else:
            clean += 1
            text.insert(END, "X=%s, Predicted = %s" % (test[i], 'Transaction Contains Cleaned Signatures') + "\n\n")

def graph():
    height = [total, clean, attack]
    bars = ('Total Transactions', 'Normal Transaction', 'Fraud Transaction')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Credit Card Fraud Detection Using AdaBoost and Majority Voting')
title.config(bg='greenyellow', fg='dodger blue')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Credit Card Dataset", command=upload)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)

modelButton = Button(main, text="Generate Train & Test Model", command=generateModel)
modelButton.place(x=350, y=550)
modelButton.config(font=font1)

runAdaBoostButton = Button(main, text="Run AdaBoost Algorithm", command=runAdaBoost)
runAdaBoostButton.place(x=650, y=550)
runAdaBoostButton.config(font=font1)

runVotingButton = Button(main, text="Run AdaBoost and Majority Voting", command=runVoting)
runVotingButton.place(x=950, y=550)
runVotingButton.config(font=font1)

predictButton = Button(main, text="Detect Fraud From Test Data", command=predicts)
predictButton.place(x=50, y=600)
predictButton.config(font=font1)

graphButton = Button(main, text="Clean & Fraud Transaction Detection Graph", command=graph)
graphButton.place(x=350, y=600)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=770, y=600)
exitButton.config(font=font1)

main.config(bg='LightSkyBlue')
main.mainloop()
