from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time
import matplotlib.pyplot as plt

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel, DRModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics

import numpy as np
import sys
    # sys.path.append('../')
    # import indic
def visualization(path = None):
    # path = 'logs/dr/2020_9_24/9/'
    # data_in = np.load(path + 'train_epoch000001input.npy')
    print("path :", path)
    if path != None:
        embed = np.load(os.path.join(path, 'embeddings.npy') )
        labels = np.load(os.path.join(path, 'labels.npy'))
        # data_la = np.load(path + 'train_epoch000001label.npy')

        # output = model(features, adj)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        # embed = output.detach().cpu().numpy()
        # label = labels.cpu().numpy()
        print('embed :', embed.shape)
        scat = ax.scatter(embed[:, 0],embed[:, 1],c=labels,s=5,cmap='rainbow')
        plt.tight_layout()
        
        plt.savefig(os.path.join(path, 'vis.png'), dpi=400)
        plt.close()
if __name__ == '__main__':

    dt = datetime.datetime.now()
    date = f"{dt.year}_{dt.month}_{dt.day}"
    path = os.path.join(os.environ['LOG_DIR'], 'dr', date)
    path = os.path.join(path, '10')
    visualization(path = path)
