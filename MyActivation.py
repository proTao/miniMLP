from abc import ABCMeta, abstractmethod, abstractproperty
from numpy import vectorize

import math


# define interface of Activation function
class Activation:
    __metaclass__=ABCMeta
    @abstractmethod
    def forward(self):
        pass
    @abstractmethod
    def backward(self):
        pass

# an inplement of Avtivation
class Relu(Activation):
    def forward(self):
        def inner(x):
            if x >= 0:
                return x
            else:
                return 0

        return vectorize(inner)

    def backward(self):
        def inner(y, cache_z):
            if cache_z >= 0:
                return 1
            else:
                return 0

        return vectorize(inner)

# an inplement of Avtivation
class LRelu(Activation):
    def __init__(self, leaky):
        self.leaky = leaky
    def forward(self):
        def inner(x):
            if x >= 0:
                return x
            else:
                return self.leaky*x

        return vectorize(inner)

    def backward(self):
        def inner(y, x):
            if x >= 0:
                return 1
            else:
                return self.leaky

        return vectorize(inner)

# an inplement of Avtivation
class Sigmoid(Activation):
    def forward(self):
        def inner(x):
            try:
                result = 1 / (1 + math.exp(-x))
            except Exception as e:
                result = 0.00000000001
            return result

        return vectorize(inner)

    def backward(self):
        def inner(y, x):
            try:
                result = 1 / (1 + math.exp(-x))
            except Exception as e:
                result = 0.00000000001
            return result*(1-result)

        return vectorize(inner)
