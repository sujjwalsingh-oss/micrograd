import random

from engine import Value


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    
        #zip means pair the self.w and x mean weight and input always keep them in pair.
        #i here in xi and wi stands for initial.
        return act.relu() if self.nonlin else act
    def parameters(self):
        return self.w + [self.b]
        #here b is single input and weight is w1,w2,w3
    
class Layer(Module):
    def __init__(self,nin,nout,**kwargs):
    # in kwargs this is only name where this means tyhe whole of nonlin true or false.
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self , x):
        out = [n(x)  for n in self.neurons]
        #give x input to every neuron mean n(x).
        return out [0] if len(out) == 1 else out

    def parameters(self):
        return [ p for n in self.neurons for p in n.parameters()]
        #n.parameters = weight = bias
    

    # now in neuron vs layer 
    #in neuron one calculator but in layer multiple outputs so multi calculator.

class MLP(Module):
#MLP stands for multi layer perceptron,it is neural netwrok made by multi layer of neurons.
    def __init__(self,nin,nouts):
        #nouts means how many neurons each layer contain
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i],sz [i + 1],nonlin = i != len(nouts) - 1 )
            for i in range (len(nouts))
        ]

    def __call__ (self , x) :
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
  # MLP
# ↓
#Layers
# ↓
#Neurons
 #↓
#Weights + Bias
# ↓
#Value objects
# ↓
#engine.py autograd
# ↓
#gradients
# ↓
#gradient descent
# ↓
#learning

    



        
        





