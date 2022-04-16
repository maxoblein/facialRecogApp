# custom l1Dist layer module
#needed to load a custom model

#import dpendeicies
import tensorflow as tf
from tensorflow.keras.layers import Layer

#custom layer 
class L1Dist(Layer):
    def __init__(self,**kwargs):
        super().__init__()
    #where it happens
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)