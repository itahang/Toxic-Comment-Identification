import numpy as np


def sigmoid(x):
    return 1.0/(1+np.exp(-x))

class LSTM():
    def __init__(self,X_t,n_neurons):
        self.T         = max(X_t.shape)
        self.X_t       = X_t
        self.n_neurons=n_neurons
        
        # Forget Gate
        
        self.bf = np.random.randn(n_neurons,1)
        self.Wf = np.random.randn(n_neurons,n_neurons+1)
        
        # Input Gate
        
        self.Wi = np.random.randn(n_neurons,n_neurons+1)
        self.bi = np.random.randn(n_neurons,1)


        self.Wc = np.random.randn(n_neurons,n_neurons+1)
        self.bc = np.random.randn(n_neurons,1)

        # Output gate

        self.Wo = np.random.randn(n_neurons,n_neurons+1)
        self.bo = np.random.randn(n_neurons,1)

        # initital hidden and cell state
        
        self.h_t_1 = np.zeros((n_neurons,1))
        self.c_t_1 = np.zeros((n_neurons,1))
        
        
        
        
    def forward(self,x_t,h_t_1,c_t_1):
        xf = np.concatenate([h_t_1,x_t],axis=0)
        
        # How much amount to forget
        
        ft = sigmoid(np.dot(self.Wf,xf) + self.bf)
        
        # Input
        
            # Information 
        it = sigmoid(np.dot(self.Wi,xf)+self.bi)
            # How much info to add
        c_hat_t = np.tanh( np.dot(self.Wc,xf)+self.bc)
        
        C_t = ft*c_t_1 + it *c_hat_t
        
        # Output gate
        # This output will be based on our cell state, 
        # but will be a filtered version
        ot = sigmoid(np.dot(self.Wo,xf)+self.bo) # 
        ht = ot*np.tanh(C_t)
        
        return ht,C_t
        
        
        
        