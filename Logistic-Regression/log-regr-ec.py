import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self):
        self.loss = []
        self.weights = None

    def sigmoid(self,val):
        return 1/(1+np.exp(-val))

    def fit(self,X,y,lr = None,epochs = 100,variance=None,estimation="MLE"):
        X = np.append(np.ones((X.shape[0],1)),X,axis=1)
        if estimation == "MLE":
            self.weights = np.ones(X.shape[1])
        else:
            self.weights = np.random.normal(loc = 0.0,scale=np.sqrt(variance),size=(X.shape[1],))
        for i in range(epochs):
          indexes = random.sample(range(len(X)), len(X))
          for j in indexes:
              x = X[j]
              #print(self.weights)
              y_hat = self.sigmoid(np.dot(x,self.weights))
              self.weights += (lr(i)*np.dot(y.iloc[j]-y_hat,x))
          y_pred = self.predict(X_train)
          self.loss.append(self.calculate_loss(y,y_pred))
        if(estimation != "MLE"):
            self.weights += (lr(i)*np.dot(y.iloc[j]-y_hat,x))+ self.weights/variance

    def predict(self,X):
        x = np.append(np.ones((X.shape[0],1)),X,axis=1)
        y = self.sigmoid(np.dot(x, self.weights))
        y[y>0.5] = 1
        y[y<0.5] = 0
        return y

    def calculate_loss(self,y_act,y_pred):
        first_term = y_act * np.log(y_pred+0.00000001)
        second_term = (1-y_act) * np.log(1-y_pred+0.00000001)
        return -np.mean(first_term + second_term)

    def calculateError(self, actual, predicted):
        return 1 - (np.sum(actual == predicted) / len(actual))

if __name__ == "__main__":
    X_train = pd.read_csv('bank-note/train.csv', header=None)
    X_test = pd.read_csv('bank-note/test.csv', header=None)
    y = X_train.iloc[:, 4]
    X_train = X_train.iloc[:, :4]
    y_test = X_test.iloc[:, 4]
    X_test = X_test.iloc[:, :4]
    variances = [0.01,0.1,0.5,1,3,5,10,100]
    T = [x for x in range(100)]
    gamma = 1
    width = 5
    learning_rate = lambda i: gamma / (1 + (gamma * i) / width)
    print("----------MAP Estimation---------")
    for variance in variances:
        print("Variance: ",variance)
        lr = LogisticRegression()
        lr.fit(X_train,y,learning_rate,variance = variance,estimation="MAP")
        predicted_test = lr.predict(X_test)
        predicted_train = lr.predict(X_train)
        print("test error: ",lr.calculateError(y_test,predicted_test))
        print("training error: ",lr.calculateError(y,predicted_train))
        fig1, ax1 = plt.subplots()
        ax1.plot(T, lr.loss, color='blue', label="training loss")
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title(f"Epochs vs Loss (Variance: {variance})")
        ax1.legend()

    print("----------MLE Estimation---------")
    for variance in variances:
        print("Variance: ",variance)
        lr = LogisticRegression()
        lr.fit(X_train,y,learning_rate,variance = variance,estimation="MLE")
        predicted_test = lr.predict(X_test)
        predicted_train = lr.predict(X_train)
        print("test error: ",lr.calculateError(y_test,predicted_test))
        print("training error: ",lr.calculateError(y,predicted_train))
        fig1, ax1 = plt.subplots()
        ax1.plot(T, lr.loss, color='blue', label="training loss")
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title(f"Epochs vs Loss (Variance: {variance})")
        ax1.legend()