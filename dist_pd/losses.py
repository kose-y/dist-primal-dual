import numpy as np
import tensorflow as tf
class LossFunction:
    @classmethod
    def eval(self,bhat, b):
        pass
    @classmethod
    def eval_deriv(self,bhat, b):
        grads = tf.gradient(self.eval(bhat, b), bhat)
        return grads[0]

class QuadLoss(LossFunction):
    @classmethod 
    def eval(self, bhat, b):
        return 0.5 * tf.norm(bhat-b)**2
    @classmethod
    def eval_deriv(self, bhat, b):
        print(bhat.shape)
        print(b.shape)
        return bhat - b

class LogisticLoss(LossFunction):
    @classmethod
    def eval(self, bhat, b):
        return tf.sum(tf.log(1+tf.exp(-b*bhat))) # when labeling is (-1, +1)
    @classmethod
    def eval_deriv(self, bhat, b):
        expbAx = tf.exp(b*bhat)
        return -b/(1.+expbAx)
    
