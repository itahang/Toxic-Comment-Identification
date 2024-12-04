import jax
from jax import numpy as jnp 
from jax import jit,grad 
from functools import partial
import functools


class LSTM():
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
        
        # Properly handle random key splitting
        keys = jax.random.split(rng, 13)  # Split into 13 keys for all parameters
        
        # Initialize weights and biases for input gate
        self.Wii = jax.random.normal(keys[0], ( hidden_size,input_size)) * 0.01
        self.Whi = jax.random.normal(keys[1], (hidden_size, hidden_size)) * 0.01
        self.bi = jax.random.normal(keys[2], (hidden_size,)) * 0.01
        
        # Initialize weights and biases for forget gate
        self.Wif = jax.random.normal(keys[3], (hidden_size,input_size )) * 0.01
        self.Whf = jax.random.normal(keys[4], (hidden_size, hidden_size)) * 0.01
        self.bf = jax.random.normal(keys[5], (hidden_size,)) * 0.01
        
        # Initialize weights and biases for cell gate
        self.Wig = jax.random.normal(keys[6], ( hidden_size,input_size)) * 0.01
        self.Whg = jax.random.normal(keys[7], (hidden_size, hidden_size)) * 0.01
        self.bg = jax.random.normal(keys[8], (hidden_size,)) * 0.01
        
        # Initialize weights and biases for output gate
        self.Wio = jax.random.normal(keys[9], ( hidden_size,input_size)) * 0.01
        self.Who = jax.random.normal(keys[10], (hidden_size, hidden_size)) * 0.01
        self.bo = jax.random.normal(keys[11], (hidden_size,)) * 0.01
        
        # Initialize default states (though these should typically be passed in)
        self.h_0 = jnp.zeros((hidden_size,1))
        self.c_0 = jnp.zeros((hidden_size,1))
    def params(self):
        """
        Return the parameters of the LSTM as a dictionary.
        
        Returns:
            dict: A dictionary containing the weights and biases of the LSTM.
                  Keys are:
                    - 'Wii': Weight matrix for input gate (input to hidden)
                    - 'Whi': Weight matrix for input gate (hidden to hidden)
                    - 'bi': Bias vector for input gate
                    - 'Wif': Weight matrix for forget gate (input to hidden)
                    - 'Whf': Weight matrix for forget gate (hidden to hidden)
                    - 'bf': Bias vector for forget gate
                    - 'Wig': Weight matrix for cell gate (input to hidden)
                    - 'Whg': Weight matrix for cell gate (hidden to hidden)
                    - 'bg': Bias vector for cell gate
                    - 'Wio': Weight matrix for output gate (input to hidden)
                    - 'Who': Weight matrix for output gate (hidden to hidden)
                    - 'bo': Bias vector for output gate
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
        Perform Single Timestamp Forward pass
        Args:
            params (dict): Paramater of the LSTM
            x_t ((input_size,1)): Input at time `t`
            c_t ((hidden_size,1)): Long Term Memory 
            h_t ((hidden_size,1)): Short Term Memory

        Returns:
            c_t_new, h_t_new
        """
        X_t = x_t.reshape(-1, 1)
        
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
        
        i = jax.nn.sigmoid(x_proj['i'] + h_proj['i'] + params['bi'].reshape(-1, 1))
        f = jax.nn.sigmoid(x_proj['f'] + h_proj['f'] + params['bf'].reshape(-1, 1))
        g = jax.nn.tanh(x_proj['g'] + h_proj['g'] + params['bg'].reshape(-1, 1))
        
        c_t_new = f * c_t + i * g
        
        o = jax.nn.sigmoid(x_proj['o'] + h_proj['o'] + params['bo'].reshape(-1, 1))
        h_t_new = o * jax.nn.tanh(c_t_new)
        
        return c_t_new, h_t_new
    forward.__doc__ = forward.__doc__
    
    @staticmethod
    def FullforwardPass(params,x,c_0,h_0,forward):
        """
        Perform Forward pass for all inputs 
        Args:
            params (dict): Paramater of the LSTM
            x_t ((input_size,1)): (batch,input_size,1)
            c_t ((hidden_size,1)): Long Term Memory 
            h_t ((hidden_size,1)): Short Term Memory
            forward : forward function of the model
        Returns:
            (c_t,h_t),h_t
        """
        def f(carry,x):
            c_t,h_t = carry
            c_t,h_t= forward(params,x,c_t,h_t)
            return (c_t,h_t),h_t
        return jax.lax.scan(f,(c_0,h_0),x)
    
    @partial(jit,static_argnums=(4))
    def forwardPass(params,x,c_0,h_0,forward):
        """
        Perform Forward pass for all inputs 
        Args:
            params (dict): Paramater of the LSTM
            x_t ((input_size,1)): (batch,input_size,1)
            c_t ((hidden_size,1)): Long Term Memory 
            h_t ((hidden_size,1)): Short Term Memory
            forward : forward function of the model
        Returns:
            h_t[-1] : Last output of the forward pass
        """   
        (_,_),f = LSTM.FullforwardPass(params,x,c_0,h_0,forward)
        return f[-1]
    forwardPass.__doc__=forwardPass.__doc__
    
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