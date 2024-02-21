"""
Created on Sat Nov 25 20:24:22 2023

@author: Yousef
"""
import numpy as np

from tkinter import *

import time

import random


root = Tk()


root.configure(background='white')

canvas = Canvas(root, width = 1400, height = 700, highlightthickness=50)

canvas['highlightbackground']= "white"



canvas.configure(bg='white')

canvas.pack()



Input = [[1,1],[2,3],[3,3],[4,3],[3,1],[-1,-1],[-2,-3],[-3,-3],[-4,-3],[-3,1]]
label = [[1],[1],[1],[1],[1],[0],[0],[0],[0],[1]]

def splitInput(Input,label):
    A = []
    B = []
    for i in range(len(Input)):
        if label[i][0] == 0:
            B.append(Input[i])
        else:
            A.append(Input[i])
            
    return (A,B)

splitted = splitInput(Input, label)

A = splitted[0]
B = splitted[1] 

print(A,B)

def constructLine(w1,w2,b, color):
    if w1 == 0:
        point1 = [0,350*(b-(650*w1)/w2)/650]
        point2 = [0,-350*(b-(650*w1)/w2)/650]
    elif w2 == 0:
        point1 = [650,0]
        point2 = [-650,0]
    else:
        point1 = [650,350*(b-(650*w1)/w2)/650]
        point2 = [-650,-350*(b-(650*w1)/w2)/650]
    #print(point1,point2)
    return canvas.create_line(-point1[0]+750,point1[1]+400,-point2[0]+750,point2[1]+400,fill=color)


xAxis = canvas.create_line(100, 400, 1400, 400)
yAxis = canvas.create_line(750, 50, 750, 750)


def drawPoints(points, color, size, expandRate):
    L = []
    for i in points:
        c = [i[0]*(expandRate),i[1]*(-(7*expandRate/13))]
        L.append(canvas.create_oval(c[0]+750-size,c[1]+400-size,c[0]+750+size,c[1]+400+size, fill = color))
    return L



h = drawPoints(A, "blue", 10, 25)
s = drawPoints(B, "red", 10, 25)   

class Layer:
    
    def __init__(self, nInputs, nNeurons, dim, name,  learnRate = 0.01):
        self.name = name
        self.dim = dim
        self.learnRate = learnRate
        if (self.dim < 2):
            self.weights = 0.01*np.random.randn(nNeurons,nInputs).T
        else:
            self.weights = 0.01*np.random.randn(nNeurons,dim,nInputs).T
        self.biases = np.zeros((1,nNeurons))
    def computeZ(self, Input):
        self.Input = Input
        return np.dot(Input,self.weights)+self.biases
    def forward(self,Input):
        output = np.tanh(self.computeZ(Input))
        if self.dim == 2:
            self.output = []
            for i in range(len(output)):
                self.output.append(output[i][0])
        else:
            self.output = output
        
        self.output = np.array(self.output)
        
    def finalize(self, Input):
        self.output = 1/(1+np.exp(-(self.computeZ(Input))))
    def predict(self):
         self.pred = np.exp(self.output)/(np.exp(self.output)+np.exp(1-self.output))
    def computeLoss(self, output):
        loss = np.square(self.pred - output)
        self.loss = 0
        for i in range(len(loss)):
            self.loss += loss[i][0]
    def update(self, dz):
        #print(self.name,"\n\n")
        #print(np.sum(dz.T, axis = 1, keepdims=True).shape,"\n\n")
        #print(np.sum(dz.T, axis = 1, keepdims=True), "\n\n")
        #print(self.Input.shape,'\n\n')
        #print(self.weights.shape,'\n\n')
        #print(self.biases.shape,"\n\n")
        #print(self.biases)
        self.weights -= (self.learnRate * (np.dot(dz.T,self.Input)).T)
        self.biases -= self.learnRate * (np.sum(dz.T, axis = 1, keepdims=True)).T
        return np.dot(self.weights, dz.T).T
       
    
    
    
def createModel(nlayers,nNeurons,learnRate=0.01):
    model = []
    for i in range(nlayers):
        if i == 0:
            model.append(Layer(1,nNeurons,2, "layer"+str(i+1),learnRate))
        elif i != nlayers-1:
            model.append(Layer(nNeurons,nNeurons,1, "layer"+str(i+1),learnRate))
        else:
            model.append(Layer(nNeurons,1,1, "layer"+str(i+1),learnRate))
    return model




    
def trainModel(model):
    lines = []
    for e in range(100):
        root.update()
        time.sleep(0.000001)
        for i in lines:
            canvas.delete(i)
        for i in range(len(model)):
            if i == 0:
                model[i].forward(np.array(Input))
            elif i != len(model)-1:
                model[i].forward(model[i-1].output)
            else:
                model[i].finalize(model[i-1].output)
                
        model[-1].predict()
        model[-1].computeLoss(np.array(label))
        #print(model[-1].output,"\n\n")
        #print(model[-1].pred, "\n\n")
        #print(np.array(label),"\n\n")
        #print("Loss =",model[-1].loss,"\n\n")
        for i in range(len(model))[::-1]:
            if i == len(model)-1:
                value = model[i].update(model[i].pred - np.array(label))
            else:
                value = model[i].update(value*(1-np.power(model[i].output,2)))
                # newdz =  np.dot(w[1],np.dot(W[2], dZ[0]) * (1-np.power(A[2],2))) * (1-np.power(A[1],2))
                
        color = ["red","blue","black","green","gold"]
        for i in range(len(model[0].weights[0][0])):
            lines.append(constructLine(model[0].weights[0][0][i], model[0].weights[0][1][i], model[0].biases[0][i],color[i]))
            
        
    


def newtrial():
    model = createModel(2, 5, 0.1)
    trainModel(model)
    #print("Loss =",model[-1].loss,"\n\n")
    #print(model[0].weights,model[0].biases)

def createSample(n):
    pos = []
    neg = []
    w1 = ((-1)**(random.randint(1,2)))*random.randint(1,100)
    w2 =  ((-1)**(random.randint(1,2)))*random.randint(1,100)
    b =  ((-1)**(random.randint(1,2)))*random.randint(1,100)
    print(w1,w2,b)
    while(not ((len(pos) == n/2) and (len(neg) == n/2))):
        x = [((-1)**(random.randint(1,2)))*random.randint(1,100),((-1)**(random.randint(1,100)))*random.randint(1,100)]
        #print(w1*x[0]+w2*x[1]+b)
        if (w1*x[0]+w2*x[1]+b) > 0 :
            if len(pos) == n/2:
                continue
            pos.append(x)
        elif (w1*x[0]+w2*x[1]+b) < 0:
            if len(neg) == n/2:
                continue
            neg.append(x)
    print("NO(+) =",len(pos))
    print("NO(-) =",len(neg))
    return [pos,neg]

#points = createSample(30)
#pointsA = points[0]
#pointsB = points[1]
#points = pointsA+pointsB
        

#precptron(points,[-1,-1,-1,-1,1,1,1,1])
frame = Frame(root, bg = "white")
button1 = Button(frame, padx=40, pady=0, text="Start", bg="white", command=newtrial, width = 1)
frame.pack()
button1.pack(side=LEFT)
root.mainloop()


