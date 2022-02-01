import tensorflow as tf
import argparse
import logging
import json
import time
import os

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from core.astromer import get_ASTROMER
from core.metrics import custom_r2
from core.losses import custom_rmse
from core.data  import load_records
from time import gmtime, strftime


cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


def run(opt):
#     os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

    # Creating (--p)royect directory
    os.makedirs(opt.p, exist_ok=True)

    # Save Hyperparameters
    conf_file = os.path.join(opt.p, 'conf.json')
    varsdic = vars(opt)
    varsdic['exp_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    with open(conf_file, 'w') as json_file:
        json.dump(varsdic, json_file, indent=4)

    # Loading data
    train_batches = load_records(os.path.join(opt.data, 'train'),
                                 opt.batch_size,
                                 max_obs=opt.max_obs,
                                 msk_frac=opt.msk_frac,
                                 rnd_frac=opt.rnd_frac,
                                 same_frac=opt.same_frac,
                                 shuffle=True,
                                 sampling=False)
    valid_batches = load_records(os.path.join(opt.data, 'val'),
                                 opt.batch_size,
                                 max_obs=opt.max_obs,
                                 msk_frac=opt.msk_frac,
                                 rnd_frac=opt.rnd_frac,
                                 same_frac=opt.same_frac,
                                 shuffle=False,
                                 sampling=False)
    with strategy.scope():
        # Training ASTROMER
        model = get_ASTROMER(num_layers=opt.layers,
                            d_model=opt.head_dim,
                            num_heads=opt.heads,
                            dff=opt.dff,
                            base=opt.base,
                            dropout=opt.dropout,
                            maxlen=opt.max_obs,
                            multi_gpu=True)
        
        model.compile(optimizer='adam',
                    loss=custom_rmse,
                    metrics=[custom_r2],
                    steps_per_execution = 50)

    # CALLBACKS
    estop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=opt.patience,
                          verbose=0,
                          mode='auto',
                          baseline=None,
                          restore_best_weights=True)
    tb = TensorBoard(log_dir=os.path.join(opt.p, 'logs'),
                     write_graph=False,
                     write_images=False,
                     write_steps_per_second=False,
                     update_freq='epoch',
                     profile_batch=0,
                     embeddings_freq=0,
                     embeddings_metadata=None)

    hist = model.fit(train_batches,
                     epochs=opt.epochs,
                     validation_data=valid_batches,
                     callbacks=[estop, tb],
                     verbose=1)

    model.save(os.path.join(opt.p, 'model.h5'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--max-obs', default=200, type=int,
                    help='Max number of observations')
    parser.add_argument('--msk-frac', default=0.5, type=float,
                        help='[MASKED] fraction')
    parser.add_argument('--rnd-frac', default=0.2, type=float,
                        help='Fraction of [MASKED] to be replaced by random values')
    parser.add_argument('--same-frac', default=0.2, type=float,
                        help='Fraction of [MASKED] to be replaced by same values')

    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/alcock', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')
    parser.add_argument('--batch-size', default=512, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=10000, type=int,
                        help='Number of epochs')
    parser.add_argument('--patience', default=40, type=int,
                        help='batch size')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU to use')
                        
    # ASTROMER HIPERPARAMETERS
    parser.add_argument('--layers', default=4, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--heads', default=8, type=int,
                        help='Number of self-attention heads')
    parser.add_argument('--head-dim', default=256, type=int,
                        help='Head-attention Dimensionality ')
    parser.add_argument('--dff', default=128, type=int,
                        help='Dimensionality of the middle  dense layer at the end of the encoder')
    parser.add_argument('--dropout', default=0.1 , type=float,
                        help='dropout_rate for the encoder')
    parser.add_argument('--base', default=1000, type=int,
                        help='base of embedding')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='optimizer initial learning rate')


    opt = parser.parse_args()
    run(opt)
