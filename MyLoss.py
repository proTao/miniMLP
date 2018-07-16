from abc import ABCMeta, abstractmethod, abstractproperty
from numpy import vectorize

# define interface of Loss function
class Loss:
    __metaclass__=ABCMeta
    @abstractmethod
    def calcLoss(self):
        pass
    @abstractmethod
    def derivation(self):
        pass

# an implement of Square
class SquareLoss(Loss):
    def calcLoss(self):
        def inner(y_hat, y):
            return (y_hat - y) ** 2 * 0.5

        return vectorize(inner)

    def derivation(self):
        def inner(y_hat, y):
            return y_hat - y

        return vectorize(inner)

