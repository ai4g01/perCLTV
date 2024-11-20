import os
import shutil

import networkx as nx
import numpy as np
import pandas as pd
import spektral
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split

from datetime import datetime
import matplotlib.pyplot as plt

from src.model import perCLTV

##############################
seed_value = 2023
lr = 0.0001
epochs = 500
beta1 = 0.5
beta2 = 0.5
timestep = 10
maxlen = 64
##############################


def plot_loss(df_hist, run_id, kfold):
    plt.plot(df_hist.index, df_hist['loss'], label='train_loss')
    plt.plot(df_hist.index, df_hist['val_loss'], label='val_loss')
    
    plt.title(f"Training and Validation Loss - {run_id}-{kfold}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"./output/ckpts/{run_id}/loss-{kfold}.png")


def data_process(timestep=10, maxlen=64):
    df_S = pd.read_csv('./data/sample_data_individual_behavior.csv')
    df_G = pd.read_csv('./data/sample_data_social_behavior.csv')
    df_Y = pd.read_csv('./data/sample_data_label.csv')

    churn_behavior_set = list(map(str, [4, 5, 7,  8, 13, 14, 16, 20, 21, 24, 29,
                              30, 34, 36, 40, 45, 49, 50, 52, 54, 55, 64, 68, 70, 73, 74, 76, 85, 87, 89]))
    payment_behavior_set = list(
        map(str,  [1, 5, 25, 26, 29, 35, 44, 46, 48, 52, 55, 56, 70, 78, 81]))

    B = df_S['seq'].apply(lambda x: x.split(
        ',') if pd.notna(x) else []).tolist()
    C = [list([xx for xx in x if xx in churn_behavior_set]) for x in B]
    P = [list([xx for xx in x if xx in payment_behavior_set]) for x in B]

    B = tf.keras.preprocessing.sequence.pad_sequences(sequences=B,
                                                      maxlen=maxlen,
                                                      padding='post')
    C = tf.keras.preprocessing.sequence.pad_sequences(sequences=C,
                                                      maxlen=maxlen,
                                                      padding='post')
    P = tf.keras.preprocessing.sequence.pad_sequences(sequences=P,
                                                      maxlen=maxlen,
                                                      padding='post')
    B = B.reshape(-1, timestep, maxlen)
    C = C.reshape(-1, timestep, maxlen)
    P = P.reshape(-1, timestep, maxlen)

    G = nx.from_pandas_edgelist(df=df_G,
                                source='src_uid',
                                target='dst_uid',
                                edge_attr=['weight'])
    A = nx.adjacency_matrix(G)
    A = spektral.layers.GATConv.preprocess(A).astype('f4')
    y1 = df_Y['churn_label'].values.reshape(-1, 1)
    y2 = np.log(df_Y['payment_label'].values + 1).reshape(-1, 1)

    print('B:', B.shape)
    print('C:', C.shape)
    print('P:', P.shape)
    print('G:', A.shape)
    print('y1:', y1.shape, 'y2:', y2.shape)

    return B, C, P, A, y1, y2


def main():
    B, C, P, A, y1, y2 = data_process(timestep=timestep, maxlen=maxlen)
    N = A.shape[0]

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value)
    run_id = datetime.now().strftime('%y%m%d%H%M%S')

    df_list = [ ]
    for kfold, item in enumerate(kfold.split(B, y1)):
        train_index, test_index = item
        train_index, val_index = train_test_split(
            train_index, test_size=0.1, random_state=seed_value)

        mask_train = np.zeros(N, dtype=bool)
        mask_val = np.zeros(N, dtype=bool)
        mask_test = np.zeros(N, dtype=bool)
        mask_train[train_index] = True
        mask_val[val_index] = True
        mask_test[test_index] = True

        # 覆盖文件方式，仅保留最好的一个模型
        checkpoint_path = f'./output/ckpts/{run_id}/ckpt-{kfold}.weights.h5'
        # checkpoint_path = f'./output/ckpts/{run_id}/ckpt-{kfold}-' + '{epoch:04d}-{val_loss:.4f}.weights.h5'
        # checkpoint_dir = os.path.dirname(checkpoint_path)

        # if os.path.exists(checkpoint_dir):
        #     shutil.rmtree(checkpoint_dir)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=10,
                                                        mode='min')

        best_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                            monitor='val_loss',
                                                            verbose=1,
                                                            save_best_only=True,
                                                            save_weights_only=True,
                                                            mode='auto')

        model = perCLTV(timestep=timestep, behavior_maxlen=maxlen)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss={'output_1': tf.keras.losses.BinaryCrossentropy(),
                            'output_2': tf.keras.losses.MeanSquaredError()},
                    loss_weights={'output_1': beta1, 'output_2': beta2},
                    metrics={'output_1': tf.keras.metrics.AUC(),
                            'output_2': 'mae'})

        hist = model.fit([B, C, P, A], [y1, y2],
                validation_data=([B, C, P, A], [y1, y2], mask_val),
                sample_weight=mask_train,
                batch_size=N,
                epochs=epochs,
                shuffle=False,
                callbacks=[early_stopping, best_checkpoint],
                verbose=1)
        
        
        df_hist = pd.DataFrame(hist.history)
        df_hist['run_id'] = f"{run_id}-{kfold}"
        df_hist.reset_index(inplace=True, drop=False, names=['epoch'])
        df_list.append(df_hist)
        
        plot_loss(df_hist, run_id, kfold)
        eval_result = model.evaluate([B, C, P, A], [y1, y2], 
                                     sample_weight=mask_test,
                                     batch_size=N,
                                     return_dict=True)
        print(eval_result)
        break
    
    df_hist_all = pd.concat(df_list)
    df_hist_all.to_csv(f'./output/ckpts/{run_id}/history.csv', index=False)


if __name__ == "__main__":
    main()