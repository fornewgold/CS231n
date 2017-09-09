import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  import math
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores_exp = np.zeros_like(scores)
    prob = np.zeros_like(scores)
    for j in xrange(num_classes):
      scores_exp[j] = math.exp(scores[j])  
    scores_exp_sum = np.sum(scores_exp)
    prob = scores_exp / scores_exp_sum
    for j in xrange(num_classes):
      dW[:, j] += (prob[j] - (j == y[i])) * X[i]
    loss_i = - math.log(prob[y[i]])
    loss += loss_i
    
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #print('function #1:')
  #print('loss =', loss)
  #print('grad =', dW)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  loss_i = np.zeros(num_train)

  scores = X.dot(W)
  scores -= np.max(scores, axis = 1, keepdims = True) # shift scores
  exp_scores = np.exp(scores)
  sum_exp_scores = np.sum(exp_scores,axis = 1)
  prob = exp_scores / sum_exp_scores[:,np.newaxis]
  loss_i = prob[np.arange(num_train), y]
  loss_i = - np.log(loss_i)
  loss = np.sum(loss_i) / num_train + 0.5 * reg * np.sum(W * W)
  
  # dW = p - 1 if y[i] = predicated class
  # dW = p otherwise
  prob[np.arange(num_train), y] -= 1
  dW = (X.T).dot(prob) / num_train + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

