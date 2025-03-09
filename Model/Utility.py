import jax 
from jax import numpy as jnp 
import optax
import typing
from typing import Callable
import functools
from .Preprocess import DSet
from sklearn.metrics import f1_score,precision_recall_curve

def bce_loss(logits:jnp.array, labels:jnp.array)->jnp.array:
    """
    Calculates BCE loss
    
    Args:
        logits: Output of Model (No sigmoid function is applied)
        labesl: Correct outputs
    
    Returns:
        BCE loss of labels
    
    """
    # Calculate per-class BCE loss
    loss = optax.sigmoid_binary_cross_entropy(logits.squeeze(), labels.squeeze())
    # Apply class weights
    return jnp.mean(loss)

def loss_fn(params:dict,lx:jnp.array,models:tuple,vmappedForward: Callable,embeddingMatrix:jnp.array,labels:jnp.array)->jnp.array:
    """
    Calculates the loss
    
    Args:
        params: Paramater of the model
        lx: input vector of form (batch,embeddings)
        models: Models
        vmappedForward: vmapped Forward function
        embeddingMatrix: Embedding Matrix
        labels: Correct labels 
    Note: This function is not Jited
    Returns:
        Loss value after applying BCE 
    """
    
    logits = vmappedForward(params,embeddingMatrix,lx,models)
    labels=jnp.expand_dims(labels,-1)
    # print(logits.shape)
    return bce_loss(logits,labels)

jit_loss_fn = jax.jit(loss_fn,static_argnums=(2,3))



def getItems(ds:DSet, start:int, end:int):
    """
    Returns rows from DSet
    
    Args:
        ds: DSet object
        start: Starting Row number
        end: Final Row number
    Note:
        Its Closed Open interval i.e [Start,end)

    Returns:
        Jax Array consisting of Tokens and Labels
    """    
    assert start>=0 and end>start ,"`Start` should be greater then zero and `End` should be striclty greater then `Start`"
    
    text, labels = [], []
    for i in range(start, end):
        # Extract the text and labels from the dataset
        t, l = ds[i]
        # Append them to the lists
        text.append(t)
        labels.append(l)
    
    # Convert to jax arrays before returning
    return jnp.array(text), jnp.array(labels)

def bestThresholdandScore(true_label:jnp.array,predicted_label:jnp.array):
    """
    Returns the best threshold where we will get the Max F1 score
    
    Args:
        true_label: Array of true labels (jnp.array)
        predicted_label: Array of true labels (jnp.array)
    
    Returns:
        best_threshold(jnp.array) , best_f1_score(jnp.array)
    """
    
    precision, recall, thresholds = precision_recall_curve(true_label, predicted_label)

    f1_scores = jnp.zeros_like(precision)

    # Only compute F1 score where precision + recall is not zero (to avoid division by zero)
    valid_indices = (precision + recall) > 0
    f1_scores=f1_scores.at[valid_indices].set( 2 * (precision[valid_indices] * recall[valid_indices]) / (precision[valid_indices] + recall[valid_indices]))

    # Find the threshold with the highest F1 score
    best_index = jnp.argmax(f1_scores)
    best_threshold = thresholds[best_index] if best_index < len(thresholds) else 0
    best_f1 = f1_scores[best_index]

    return best_threshold,best_f1