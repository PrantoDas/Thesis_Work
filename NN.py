import numpy as np

class ActivationFuntion:
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    
    def sigmoid_backward(self,dA, Z):
        s = self.sigmoid(Z)
        return dA*s*(1-s)
    
    def relu(self,Z):
        return np.maximum(0,Z)
    
    def relu_backward(self,dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

class NeuralNetwork(ActivationFuntion):
    def __init__(self, layer_d, Y, learning_rate):
        self.L = len(layer_d)-1
        self.Y = Y
        self.learning_rate = learning_rate
        self.values = {}
        for l in range(1,self.L+1):
            self.values['W'+str(l)] = np.random.randn(layer_d[l],layer_d[l-1])/np.sqrt(layer_d[l-1])
            self.values['b'+str(l)] = np.zeros((layer_d[l],1))
    
    def calculate_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (1.0/m)*(-np.dot(Y,np.log(AL).T)-np.dot(1-Y,np.log(1-AL).T))
        return np.squeeze(cost)
    
    def test(self, X, Y, cal_Cost = True):
        A = X
        for l in range(1,self.L):
            Z = self.values['W'+str(l)].dot(A)+self.values['b'+str(l)]
            A = self.relu(Z)
        
        Z = self.values['W'+str(self.L)].dot(A)+self.values['b'+str(self.L)]
        A = self.sigmoid(Z)
        
        m = X.shape[1]
        if(cal_Cost): cost = self.calculate_cost(A,Y)
        
        p = np.zeros((1,m))
        for i in range(0, A.shape[1]):
            if A[0,i] > 0.5 :
                p[0,i] = 1
            else:
                p[0,i] = 0
                
        accuracy = np.sum((p == Y)/m)
        
        if(cal_Cost): return p, cost, accuracy
        return p, accuracy
        
                   
    def train(self, l, A_prev):
        W = self.values['W'+str(l)]
        b = self.values['b'+str(l)]
        Z = W.dot(A_prev)+b
        
        if l == self.L:
            A = self.sigmoid(Z)
            Y = self.Y
            dA = -(np.divide(Y, A)-np.divide(1-Y,1-A))
            dZ = self.sigmoid_backward(dA, Z)
            
        else:
            A = self.relu(Z)
            dA = self.train(l+1, A)
            dZ = self.relu_backward(dA, Z)
            
        m = A_prev.shape[1]           
        dW = 1.0/m*np.dot(dZ, A_prev.T)
        db = 1.0/m*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T,dZ)
        
        self.values['W'+str(l)] -= self.learning_rate*dW
        self.values['b'+str(l)] -= self.learning_rate*db
        return dA_prev
