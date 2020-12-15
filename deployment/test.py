import tensorflow as tf 
from tensorflow import keras
import numpy as np
from model import get_G 

input_g_shape =(96,96,3)
G = get_G(input_g_shape)
G.load_weights("g_maevgg.h5")
G.summary()



