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
  for i in xrange(X.shape[0]):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    l_i = -scores[y[i]]
    denominator = np.sum(np.exp(scores))

    for c_idx in xrange(W.shape[1]):
      if c_idx == y[i]:
        dW[:, c_idx] += (np.exp(scores[c_idx])/denominator - 1)*X[i]
      else:
        dW[:, c_idx] += np.exp(scores[c_idx])/denominator*X[i]
    l_i += np.log(denominator)
    loss += l_i

  loss /= X.shape[0]
  loss += reg * np.sum(W*W)
  dW /= X.shape[0]
  dW += 2 *reg * W

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
  scores = X.dot(W)
  scores -= np.max(scores, axis=1)[:,np.newaxis]
  exp_scores = np.exp(scores)

  n_range = np.arange(X.shape[0])
  denominators = np.sum(exp_scores, axis=1)
  scores = exp_scores/denominators[:,np.newaxis]
  loss = np.sum(np.log(scores[n_range,y]))*-1/X.shape[0] + reg*np.sum(W*W)

  scores[n_range, y] -= 1
  dW = X.T.dot(scores) / X.shape[0] + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

