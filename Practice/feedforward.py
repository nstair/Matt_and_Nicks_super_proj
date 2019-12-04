import matplotlib.pylab as plt
import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

w1 = np.array([[0.2, 0.2, 0.2],[0.4, 0.4, 0.4],[0.6, 0.6, 0.6]])
w2 = np.array([[0.5, 0.5, 0.5]])

weights = np.array([w1, w2])

b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])

biases = np.array([b1, b2])

inpt = np.array([1.5, 2.0, 3.0])

def ff(inpt, weights, biases):

	layer = inpt
	#Object to keep track of node values in the current
	#working layer. Starts with input node array.

	#Each layer will have an array of weight arrays, which
	#themselves have weight values. For each output node
	#in the new layer, all nodes from the previous layer will
	#be multiplied by their associated weight, summed along-
	#-side that layers bias term; that sum will be run through
	#the activation function (in this case the sigmoid), and
	#that output is the value of that node in the new layer.

	#This process repeats for every node in the new layer,
	#and each new node will have its own set of weights to
	#modify the value of the previous layer, as well as an
	#associated bias term for the new node.

	#Since each layer to layer operation has its own set
	#of associated weights, we can loop through each layer
	#operation using the meta array of weights.
	for l in range(len(weights)):

		newLayer = []
		#Holder for nodes in the upcoming layer.
		#When filled, this object will overwite 
		#the 'layer' variable.

	#Loop through each set of weights in the current layer.
		for i in range(len(weights[l])):

			nodeSum = 0
			#We are now in the weight set for a single
			#node in the upcoming layer, so we can start
			#adjusting that node's value.

		#Loop through each weight, corrosponding to the
		#set of nodes in the previous layer.
			for k in range(len(weights[l][i])):
				
				#Multiply each weight by its
				#associated node in the current layer
				#and tally its sum.
				nodeSum += weights[l][i][k] * layer[k]

			#Add the bias term. Recall that every node
			#in the new layer gets a bias term added,
			#so we back out one loop from the weights
			#for every previous node into the loop
			#handling every node in the new layer.
			nodeSum += biases[l][i]

			#For every node in the new layer, we take
			#the computed sum and run it through our
			#activation function. Finally we append
			#the node to our new layer.
			newLayer.append(sigmoid(nodeSum))

		#We set our new layer of freshly computed nodes
		#and allow the function to use its next set(s)
		#of associated weight arrays to repeat the process.
		layer = newLayer

	return layer[0]

ff(inpt, weights, biases)