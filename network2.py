import numpy as np
from MyActivation import *
from MyLoss import *


# neural network layer
class Layer:

    def __init__(self, n0, n1, alpha, deep):
        """
        n0 is the number of the elements of the pre_layer
        n1 is the number of the elements of the current layer
        """
        self.deep = str(deep) # for debug and output information
        self.W = np.matrix(np.random.rand(n1, n0) * 0.1)
        self.b = np.matrix(np.random.rand(n1, 1))
        self.alpha = alpha

    def setActivationFunction(self, func):
        self.forward_func = func.forward()
        self.back_func = func.backward()

    def receiveInput(self, X):
        # the dimension of the x is (n0,m)
        # print("layer " + self.deep + " receive input")
        self.X = X

    def forwardPropagate(self):
        # the dimension of the z is (n1,1)
        # print("W'shape of Layer"+self.deep+" : "+str(self.W.shape))
        # print("X'shape of Layer"+self.deep+" : "+str(self.X.shape))
        z = self.W * self.X + self.b
        self.cache_z = z
        a = self.forward_func(z)
        return a

    def show(self):
        print("--------W-------")
        print(self.W)
        print("--------B-------")
        print(self.b)
        print("--------X-------")
        print(self.X)
        print("\n")
        

    def receiveOutputDerivation(self, da):
        self.da = da

    def backPropagate(self):
        m = self.da.shape[1]

        acti_derivation = self.back_func(self.da, self.cache_z)


        dz = np.multiply(acti_derivation, self.da) / m
        # print("da")
        # print(self.da)
        # print("dz")
        # print(dz)
        result = self.W.T * dz
        self.update(dz)
        return result

    def update(self, dz):
        m = dz.shape[1]
        self.dW = dz * self.X.T
        # print("dW")
        # print(self.dW)
        self.db = np.sum(dz,1) / m
        self.W = self.W - self.dW * self.alpha
        self.b = self.b - self.db * self.alpha




class Network(object):
    def __init__(self):
        self.__train_X_set = None
        self.__train_Y_set = None
        self.__deep = None
        self.__network = []
        self.__loss_threshold = 0.0000001
        self.__max_iter = 50000
        self.__network_parameter = None
        self.__running_mode = False
        self.__cost_function = None

    def setTrainSetX(self, X):
        self.__train_X_set = X

    def setTrainSetY(self, Y):
        self.__train_Y_set = Y

    def setLossThreshold(self, sigma):
        self.__loss_threshold = sigma

    def setMaxIterRound(self, max_iter):
        self.__max_iter = max_iter

    def constructNetwork(self, deep, parameter, inner_acti_func, output_acti_func):
        self.__network = []
        self.__deep = deep

        for i in range(len(parameter)-1):
            assert parameter[i][1] == parameter[i+1][0], "Layer %d not match Layer %d" % (i, i+1)
        assert parameter[-1][1] == 1, "the last Layer must with one output node"
        assert deep == len(parameter), "deep does not match "

        self.__network = []
        for i in range(deep):
            self.__network.append(Layer(parameter[i][0], parameter[i][1], parameter[i][2], i))
            self.__network[i].setActivationFunction(inner_acti_func)
        self.__network[deep-1].setActivationFunction(output_acti_func)

    def setCostFunction(self, cost_function):
        self.__cost_function = cost_function

    def setRunningMode(self, mode):
        self.__running_mode = mode

    def testBackProp(self, iter_round = 1):
        network_w_derivation = []
        test_w_derivation = []
        epsilon = 0.00001
        # only need one sample
        self.__network[0].receiveInput(self.__train_X_set[:,0])
        

        for i in range(self.__deep):
            # print("W")
            # print(self.__network[i].W)
            # print("b")
            # print(self.__network[i].b)
            # print("X")
            # print(self.__network[i].X)
            for j in range(self.__network[i].W.shape[0]):
                for k in range(self.__network[i].W.shape[1]):
                    # right epsilon
                    self.__network[i].W[j,k] += epsilon
                    for l in range(self.__deep - 1):
                        middle = self.__network[l].forwardPropagate()
                        self.__network[l+1].receiveInput(middle)
                    y_hat1 = self.__network[self.__deep-1].forwardPropagate()
                    loss1 = np.sum(self.__cost_function.calcLoss()(self.__train_Y_set[0,0], y_hat1))

                    # left epsilon
                    self.__network[i].W[j,k] -= epsilon * 2
                    for l in range(self.__deep - 1):
                        middle = self.__network[l].forwardPropagate()
                        self.__network[l+1].receiveInput(middle)
                    y_hat2 = self.__network[self.__deep-1].forwardPropagate()
                    loss2 = np.sum(self.__cost_function.calcLoss()(self.__train_Y_set[0,0], y_hat2))

                    self.__network[i].W[j,k] += epsilon
                    test_w_derivation.append((loss1 - loss2)/(epsilon*2))

            for j in range(self.__network[i].b.shape[0]):
                # right epsilon
                self.__network[i].b[j,0] += epsilon
                for l in range(self.__deep - 1):
                    middle = self.__network[l].forwardPropagate()
                    self.__network[l+1].receiveInput(middle)
                y_hat1 = self.__network[self.__deep-1].forwardPropagate()
                loss1 = np.sum(self.__cost_function.calcLoss()(self.__train_Y_set[0,0], y_hat1))

                # left epsilon
                self.__network[i].b[j,0] -= epsilon * 2
                for l in range(self.__deep - 1):
                    middle = self.__network[l].forwardPropagate()
                    self.__network[l+1].receiveInput(middle)
                y_hat2 = self.__network[self.__deep-1].forwardPropagate()
                loss2 = np.sum(self.__cost_function.calcLoss()(self.__train_Y_set[0,0], y_hat2))
                    
                self.__network[i].b[j,0] += epsilon
                test_w_derivation.append((loss1 - loss2)/(epsilon*2))


        # calc network derivation
        # forward propagation
        for i in range(self.__deep - 1):
            middle = self.__network[i].forwardPropagate()
            self.__network[i+1].receiveInput(middle)
            # print("middle")
            # print(middle)
        y_hat = self.__network[self.__deep-1].forwardPropagate()
        # print("yhat")
        # print(y_hat)

        # calculate loss
        loss = np.sum(self.__cost_function.calcLoss()(self.__train_Y_set[0,0], y_hat))
        loss_derivation = self.__cost_function.derivation()(y_hat, self.__train_Y_set[0,0])
        # print("loss_dev")
        # print(loss_derivation)

        # backward propagation
        self.__network[self.__deep-1].receiveOutputDerivation(loss_derivation)
        for i in range(self.__deep-1,0,-1):
            middle = self.__network[i].backPropagate()
            self.__network[i-1].receiveOutputDerivation(middle)
        self.__network[0].backPropagate()

        for i in range(self.__deep):
            network_w_derivation.extend(self.__network[i].dW.reshape(1,self.__network[i].dW.size).tolist()[0])
            network_w_derivation.extend(self.__network[i].db.reshape(1,self.__network[i].db.size).tolist()[0])
        
        # print(test_w_derivation)        
        # print(network_w_derivation)
        delta_vector=np.matrix(test_w_derivation)-np.matrix(network_w_derivation)
        print(delta_vector)

    def Train(self):
        last_loss = float('inf')
        for i in range(self.__max_iter):

            # forward propagation
            self.__network[0].receiveInput(self.__train_X_set)
            for j in range(self.__deep - 1):
                middle = self.__network[j].forwardPropagate()
                self.__network[j+1].receiveInput(middle)
            y_hat = self.__network[self.__deep-1].forwardPropagate()
            

            # calculate loss
            loss = np.sum(self.__cost_function.calcLoss()(Y, y_hat))
            if abs(loss-last_loss) < self.__loss_threshold:
                print("our loss get constriction, so break at iteration %d" % i)
                break
            last_loss = loss
            loss_derivation = self.__cost_function.derivation()(y_hat, Y)


            # backward propagation
            self.__network[self.__deep-1].receiveOutputDerivation(loss_derivation)
            for j in range(self.__deep-1,0,-1):
                middle = self.__network[j].backPropagate()
                self.__network[j-1].receiveOutputDerivation(middle)
            self.__network[0].backPropagate()
            if i%100 ==0:
                print("iteration %d loss %s \n" % (i, str(loss)))


    def Predict(self, X):
        self.__network[0].receiveInput(X)
        for j in range(self.__deep-1):
            middle = self.__network[j].forwardPropagate()
            self.__network[j+1].receiveInput(middle)
        return self.__network[self.__deep-1].forwardPropagate()


# make train set
if __name__ == "__main__":
    X = np.matrix(np.random.randint(1,50,size=(2,200)))
    fit_func = np.vectorize(lambda x,y: x*2+y)
    Y = fit_func(X[0],X[1])
    
    parameter = [[2,4,0.005],[4,4,0.005],[4,4,0.005],[4,1,0.005]]

    network = Network()
    network.setTrainSetX(X)
    network.setTrainSetY(Y)
    network.setMaxIterRound(100000)
    network.setLossThreshold(0.00000001)
    network.setRunningMode(False)
    network.constructNetwork(4, parameter, Relu(), Relu())
    network.setCostFunction(SquareLoss())

    network.Train()


    train_label=network.Predict(X)
    print(X)
    print(train_label)

    X_predict = np.matrix([[6,3,5],[7,20,1]])
    predict_label = network.Predict(X_predict)
    print(predict_label)

    # network.testBackProp()