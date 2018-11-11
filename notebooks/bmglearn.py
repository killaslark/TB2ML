import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing,datasets
import matplotlib.pyplot as plt
import sys


def sigmoid_derivate(x):
    return (x) * (1.0 - (x))

def activate_sigmoid(sum):
    return(1.0 / (1.0 + np.exp(-sum)))


class NeuralNetwork:
    def __init__(self, X, n_hidden_layers = 1, output_neurons = 1, momentum = False, learning_rate = 0.05):
        self.data = X
        self.learning_rate = learning_rate
        self.input_neurons = len(X[0])
        self.net = list()
        self.hidden_neurons = self.input_neurons + 1
        self.output_neurons = output_neurons
        
        self.n_hidden_layers = n_hidden_layers
        
    def last_layer_neuron(self):
        return len(self.net[-1])
    
    def append_layer(self, curr_neurons, input_dim=-1):
        if (input_dim == -1):
            input_dim = self.last_layer_neuron()

        hidden_layer = [ {'weights': np.random.uniform(size=input_dim)} for i in range(curr_neurons)]
        self.net.append(hidden_layer)

    def initialize_network(self):
        net = list()
        for h in range(self.n_hidden_layers):
            if h!=0:
                self.input_neurons=len(net[-1])

            hidden_layer = [ { 'weights': np.random.uniform(size=self.input_neurons)} for i in range(self.hidden_neurons) ]
            net.append(hidden_layer)
    
        output_layer = [ { 'weights': np.random.uniform(size=self.hidden_neurons)} for i in range(self.output_neurons)]
        net.append(output_layer)
        
        self.net = net
        
    def print_network(self):
        for i,layer in enumerate(self.net,1):
            print("Layer {} ".format(i))
            for j,neuron in enumerate(layer,1):
                print("neuron {} :".format(j),neuron)
        
                

    def feed_forward(self,input):
        row=input
        for layer in self.net:
            prev_input=np.array([])
            for neuron in layer:
                sum=neuron['weights'].T.dot(row)

                result=activate_sigmoid(sum)
                neuron['result']=result

                prev_input=np.append(prev_input,[result])
            row = prev_input

        return row
    
    def back_propagation(self , row, expected):
        for i in reversed(range(len(self.net))):
            layer=self.net[i]
            errors=np.array([])
            if i==len(self.net)-1:
                results=[neuron['result'] for neuron in layer]
                errors = expected-np.array(results) 
            else:
                for j in range(len(layer)):
                    herror=0
                    nextlayer=self.net[i+1]
                    for neuron in nextlayer:
                        herror+=(neuron['weights'][j]*neuron['delta'])
                    errors=np.append(errors,[herror])

            for j in range(len(layer)):
                neuron=layer[j]
                neuron['delta']=errors[j]*sigmoid_derivate(neuron['result'])
                
    def updateWeights(self,input):
        for i in range(len(self.net)):
            inputs = input
            if i!=0:
                inputs=[neuron['result'] for neuron in self.net[i-1]]

            for neuron in self.net[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j]+=self.learning_rate*neuron['delta']*inputs[j]
    
    
    def training(self, X, y, epochs= 128):
        self.data = X
        errors=[]
        for epoch in range(epochs):
            sum_error=0
            for i,row in enumerate(X):
                outputs = self.feed_forward(row)

                # asumsi output hanya 1 neuron
#                 expected = [0.0 for i in range(self.output_neurons)]
                expected = [y[i]]

                sum_error += sum([(expected[j]-outputs[j])**2 for j in range(len(expected))])
                self.back_propagation(row, expected)
                self.updateWeights(row)
            if epoch%10000 ==0:
                print('>epoch=%d,error=%.3f'%(epoch,sum_error))
                errors.append(sum_error)
        return errors
    
    def predict(self, row):
        outputs = self.feed_forward(row)
        return outputs
