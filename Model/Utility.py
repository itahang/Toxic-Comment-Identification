import jax 
from jax import numpy as jnp
from functools import partial
from NN import LSTM
import optax


@jax.jit
def stable_softmax(z):
    z_stable = z - jnp.max(z)
    exp_z = jnp.exp(z_stable)
    return exp_z / jnp.sum(exp_z)

@partial(jax.jit,static_argnums=(2))
def ModelForward(params,x,models):
    *lstm,mlp = models
    h_0=x
    for i,model in enumerate(lstm):
        (_,_),h_0=model.FullforwardPass(params[i],h_0,model.c_0,model.h_0,LSTM.forward)
    return  (mlp.forward(params[-1],h_0[-1])).squeeze()

@partial(jax.jit,static_argnums=(1))
def loss_fn(params,models,x,y):
    Y=stable_softmax(ModelForward(params,x,models)).squeeze()
    return optax.safe_softmax_cross_entropy(Y,y)