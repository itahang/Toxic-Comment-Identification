import jax
from jax import numpy as jnp 
from jax import jit,grad 
from functools import partial
import functools
from jax import random


import jax
import jax.numpy as jnp
from jax import jit

class LSTM:
    def __init__(self, input_size: int, hidden_size: int, rng):
        """
        Constructor for the LSTM class.
        -------------------------------
        Parameters:
          input_size (int): The number of input features.
          hidden_size (int): The number of units in the hidden state.
          rng (jax.random.PRNGKey): Random number generator key.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Split key into 13 for all parameters
        keys = jax.random.split(rng, 13)
        
        # Input gate weights and bias
        self.Wii = jax.nn.initializers.orthogonal()(keys[0], (hidden_size, input_size))*2.0
        self.Whi = jax.nn.initializers.orthogonal()(keys[1], (hidden_size, hidden_size))*2.0
        self.bi  = jnp.ones((hidden_size, 1))*2.0  # or use an initializer if preferred
        
        # Forget gate weights and bias
        self.Wif = jax.nn.initializers.orthogonal()(keys[3], (hidden_size, input_size))*2.0
        self.Whf = jax.nn.initializers.orthogonal()(keys[4], (hidden_size, hidden_size))*2.0
        self.bf  = jnp.ones((hidden_size, 1))*2.0
        
        # Cell gate weights and bias
        self.Wig = jax.nn.initializers.orthogonal()(keys[6], (hidden_size, input_size))*2.0
        self.Whg = jax.nn.initializers.orthogonal()(keys[7], (hidden_size, hidden_size))*2.0
        self.bg  = jnp.ones((hidden_size, 1))*2.0
        
        # Output gate weights and bias
        self.Wio = jax.nn.initializers.orthogonal()(keys[9], (hidden_size, input_size))*2.0
        self.Who = jax.nn.initializers.orthogonal()(keys[10], (hidden_size, hidden_size))*2.0
        self.bo  = jnp.ones((hidden_size, 1))*2.0
        
        # Initial states
        self.h_0 = jnp.zeros((hidden_size, 1))
        self.c_0 = jnp.zeros((hidden_size, 1))
    
    def params(self):
        """
        Return the parameters of the LSTM as a dictionary.
        """
        return {
            'Wii': self.Wii, 'Whi': self.Whi, 'bi': self.bi,
            'Wif': self.Wif, 'Whf': self.Whf, 'bf': self.bf,
            'Wig': self.Wig, 'Whg': self.Whg, 'bg': self.bg,
            'Wio': self.Wio, 'Who': self.Who, 'bo': self.bo,
        }
    
    @staticmethod
    @jit
    def forward(params, x_t, c_t, h_t):
        """
        Single time-step forward pass.
        Args:
            params (dict): LSTM parameters.
            x_t ((input_size,) or (input_size,1)): Input at time t.
            c_t ((hidden_size,1)): Previous cell state.
            h_t ((hidden_size,1)): Previous hidden state.
        Returns:
            (c_t_new, h_t_new)
        """
        X_t = x_t.reshape(-1, 1)
        
        # Compute projections for each gate
        x_proj = {
            'i': params['Wii'] @ X_t,
            'f': params['Wif'] @ X_t,
            'g': params['Wig'] @ X_t,
            'o': params['Wio'] @ X_t
        }
        
        h_proj = {
            'i': params['Whi'] @ h_t,
            'f': params['Whf'] @ h_t,
            'g': params['Whg'] @ h_t,
            'o': params['Who'] @ h_t
        }
        
        i = jax.nn.sigmoid(x_proj['i'] + h_proj['i'] + params['bi'])
        f = jax.nn.sigmoid(x_proj['f'] + h_proj['f'] + params['bf'])
        g = jax.nn.tanh(x_proj['g'] + h_proj['g'] + params['bg'])
        o = jax.nn.sigmoid(x_proj['o'] + h_proj['o'] + params['bo'])
        
        c_t_new = f * c_t + i * g
        h_t_new = o * jax.nn.tanh(c_t_new)
        
        return c_t_new, h_t_new
    
    @staticmethod
    def FullforwardPass(params, x, c_0, h_0, forward):
        """
        Full forward pass over a sequence using lax.scan.
        Args:
            params (dict): LSTM parameters.
            x (array): Sequence input of shape (seq_length, input_size) or (seq_length, input_size, 1).
            c_0 ((hidden_size,1)): Initial cell state.
            h_0 ((hidden_size,1)): Initial hidden state.
            forward: The single time-step forward function.
        Returns:
            final_carry, outputs  where final_carry = (c_final, h_final)
        """
        def f(carry, x_t):
            c_t, h_t = carry
            c_t, h_t = forward(params, x_t, c_t, h_t)
            return (c_t, h_t), h_t
        
        return jax.lax.scan(f, (c_0, h_0), x)
    
    @staticmethod
    @jit
    def forwardPass(params, x, c_0, h_0, forward):
        """
        Forward pass for all time steps, returning the final hidden state.
        Args:
            params (dict): LSTM parameters.
            x (array): Sequence input of shape (seq_length, input_size) or similar.
            c_0 ((hidden_size,1)): Initial cell state.
            h_0 ((hidden_size,1)): Initial hidden state.
            forward: The single time-step forward function.
        Returns:
            h_final: The final hidden state.
        """
        (final_c, final_h), _ = LSTM.FullforwardPass(params, x, c_0, h_0, forward)
        return final_h

class BiDirectionalLSTM:
    def __init__(self,input_size: int, hidden_size: int, rng):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Split key into 13 for all parameters
        keys = jax.random.split(rng, 2)
        self.forwardLSTM = LSTM(input_size,hidden_size,keys[0])
        self.reverseLSTM = LSTM(input_size,hidden_size,keys[1])
    
    def params(self):
        return self.forwardLSTM.params(),self.reverseLSTM.params()
    
    def forwards(self):
        return self.forwardLSTM.FullforwardPass,self.reverseLSTM.FullforwardPass
        
    @staticmethod
    @jit
    def forward(params,x,c_0,h_0):
        forwardParams,reverseParams = params 
        (_,_),forwardX=LSTM.FullforwardPass(forwardParams,x,c_0,h_0,LSTM.forward)
        (_,_),reverseX=LSTM.FullforwardPass(reverseParams,x[-1::-1,:],c_0,h_0,LSTM.forward)
        Xs=jnp.array([jnp.concat([forwardX[i,:],reverseX[i,:]],axis=1).reshape(-1,1) for i in range(x.shape[0])])
        return Xs

class MLP:
    def __init__(self,numinputs,numOuts,rng):
        self.numinputs=numinputs
        self.numOuts=numOuts
        self.W = jax.random.normal(rng,(numOuts,numinputs))
        self.b = jnp.zeros((numOuts,1))
        
    def params(self):
        """Return the parameters as a dictionary."""
        return {
            'W': self.W,  'b': self.b,
        }

    @staticmethod
    @jit
    def forward(params,x):
        x =x.reshape(-1,1)
        return params['W']@x + params['b'] 



class Dropout:
    def __init__(self, rate,rng=10):
        assert 0 < rate < 1.0, "Dropout rate should be between 0 and 1"
        self.rate = rate
        self.rng = rng
        self.key = random.PRNGKey(rng)
    
    def __call__(self, layer):
        """
        Layer should be a 1D or 2D numpy array.
        Applies dropout by setting a fraction of activations to zero.
        """
        assert len(layer.shape) in [1, 2], "Layer should be a 1D or 2D numpy array."
        mask=random.binomial(self.key,1,1-self.rate,shape=layer.shape)
        _,key=random.split(self.key)
        return layer * mask /(1-self.rate)
def maxPooling(inputs:jnp.array):
    return jnp.max(inputs,axis=0)

    