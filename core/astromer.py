import tensorflow as tf
import logging
import os, sys

from core.output    import RegLayer
from core.losses    import custom_rmse
from core.metrics   import custom_acc
from core.encoder   import Encoder
from core.metrics import custom_r2

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tqdm import tqdm

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
os.system('clear')

def get_ASTROMER(num_layers=2,
                 d_model=200,
                 num_heads=2,
                 dff=256,
                 base=10000,
                 dropout=0.1,
                 use_leak=False,
                 no_train=False,
                 maxlen=100,
                 batch_size=None,
                 multi_gpu=False):
    
#     if multi_gpu:
#         mirrored_strategy = tf.distribute.MirroredStrategy()
#     else:
#         mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
        
    serie  = Input(shape=(maxlen, 1),
              batch_size=None,
              name='input')
    times  = Input(shape=(maxlen, 1),
              batch_size=None,
              name='times')
    mask   = Input(shape=(maxlen, 1),
              batch_size=None,
              name='mask')

    placeholder = {'input':serie,
                   'mask_in':mask,
                   'times':times}
        
#     with mirrored_strategy.scope():
    encoder = Encoder(num_layers,
                d_model,
                num_heads,
                dff,
                base=base,
                rate=dropout,
                use_leak=use_leak,
                name='encoder')

    if no_train:
        encoder.trainable = False

    x = encoder(placeholder)

    x = RegLayer(name='regression')(x)

    return Model(inputs=placeholder,
                     outputs=x,
                     name="ASTROMER")

def astromer_encoder(path, trainable=False):
    astromer = tf.keras.models.load_model(os.path.join(path, 'model.h5'),
                                          custom_objects={'Encoder': Encoder,
                                          'RegLayer': RegLayer,
                                          'custom_rmse':custom_rmse,
                                          'custom_r2':custom_r2})

    astromer.trainable = trainable
    encoder = astromer.get_layer('encoder')
    return encoder
