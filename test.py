import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os


strategy = tf.distribute.get_strategy()
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync

print(REPLICAS)
print(tf.config.list_physical_devices('GPU'))

flag = os.path.exists('./22.h5')

def create_model(model_name, shape):
    with strategy.scope():
        input_layer = tf.keras.Input(shape = shape)
        construct = getattr(keras.applications, model_name)
        mid_layer = construct(include_top = False, 
                            weights = None, 
                            pooling = 'avg')(input_layer)
        last_layer = keras.layers.Dense(1, activation = 'sigmoid')(mid_layer)
        model = keras.Model(input_layer, last_layer)
    return model
def compile_new_model(model):
    with strategy.scope():
        loss = keras.losses.BinaryCrossentropy()
        optimizer = keras.optimizers.SGD()
        prec = keras.metrics.Precision(name = 'prec')
        rec = keras.metrics.Recall(name = 'rec')
        model.compile(
            loss = loss,
            optimizer = optimizer,
            metrics = [prec, rec]
        )
    return model

with strategy.scope():
    model = create_model('ResNet50', (256, 256, 3))
    model = compile_new_model(model)

print('Loading weights started.')
model.load_weights('./22.h5')
print('Weights loaded.')
