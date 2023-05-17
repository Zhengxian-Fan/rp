print('starting run....')

import sys

print('starting run....')

sys.path.insert(0,'/gpfs3/well/rahimi/users/gra027/JNb/')

from Graph.ModelPkg.BEHRTRaw import *
import torch.nn as nn
import torch
import numpy as np

from torch.utils.data.dataset import Dataset
import time
from torch.autograd import Variable
from Graph.ModelPkg import utils
from Graph.ModelPkg.MLMRaw import *

from Graph.ModelPkg.DataProc import *
import os
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import sklearn.metrics as skm
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import torch

import os
import argparse
import random

import torch
import torch.utils.data
from torch import nn, optim
from Graph.ModelPkg.pytorch_pretrained_bert  import optimizer
from torch.nn import functional as F

from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
file_config = {
        'vocab': '/gpfs3/well/rahimi/users/gra027/JNb/Graph/RP/HF/Data/GeneralVocDM',

    'fulld':    '/gpfs3/well/rahimi/users/gra027/JNb/Graph/RP/HF/Data/MLM_wY.parquet/',
    # 'fulld': '/gpfs3/well/rahimi/users/gra027/JNb/Graph/RP/HF/Data/MLM_wYsample.parquet/',
    #
    'yearVocab': '/gpfs3/well/rahimi/users/gra027/JNb/ExpHypCancer/Data/AllSubclass/year_dict_pureICD',
}


optim_config = {
    'lr': 3e-5,
    'warmup_proportion': 0.1,


}

global_params = {
    'batch_size': 256,
    'gradient_accumulation_steps': 1,
    'device': 'cuda:0',
    'output_dir':  '/gpfs3/well/rahimi/users/gra027/JNb/Graph/RP/HF/SavedM',
    'output_name': 'MLM_BEHRT_DMAY.bin',
    'save_model': True,
    'max_len_seq': 200,
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'min_visit': 5,
    'yearOn':True
}



YearVocab = utils.load_obj(file_config['yearVocab'])
create_folder(global_params['output_dir'])
BertVocab = utils.load_obj(file_config['vocab'])
print(len(BertVocab['token2idx']))
poo
ageVocab, _ = utils.age_vocab(max_age=global_params['max_age'], year=global_params['age_year'], symbol=global_params['age_symbol'])
fulldata = pd.read_parquet(file_config['fulld'])
print('read data....')

trainSet = MLMLoader(token2idx=BertVocab['token2idx'], dataframe=fulldata, max_len=global_params['max_len_seq'], max_age=global_params['max_age'], year=global_params['age_year'], age_symbol=global_params['age_symbol'])
trainload = DataLoader(dataset=trainSet, batch_size=global_params['batch_size'], shuffle=True)

model_config = {
    'vocab_size': len(BertVocab['token2idx'].keys()), # number of disease + symbols for word embedding
    'hidden_size': 150, # word embedding and seg embedding hidden size
    'seg_vocab_size': 2, # number of vocab for seg embedding
    'age_vocab_size': len(ageVocab.keys()), # number of vocab for age embedding
    'max_position_embedding': global_params['max_len_seq'], # maximum number of tokens
    'hidden_dropout_prob': 0.15, # dropout rate
    'num_hidden_layers': 4, # number of multi-head attention layers required
    'num_attention_heads': 6, # number of attention heads
    'attention_probs_dropout_prob': 0.15, # multi-head attention dropout rate
    'intermediate_size': 108, # the size of the "intermediate" layer in the transformer encoder
    'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    'initializer_range': 0.02, # parameter weight initializer range,
    'yearOn':True,
    'year_vocab_size': len(YearVocab['token2idx'].keys()),

}
conf = BertConfig(model_config)
model = BertForMaskedLM(conf)
# fullBert = os.path.join('/gpfs3/well/rahimi/users/gra027/JNb/Graph/RP/HF/SavedM/', "BEHRT_MLM_DMAY_4graphtests.bin")
# model = toLoad(model, fullBert)
model = model.to(global_params['device'])
optim = optimizer.adam(params=list(model.named_parameters()), config=optim_config)


def train(model, e, validload, optim):
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    model.train()
    for step, batch in enumerate(validload):
        batch = tuple(t.to(global_params['device']) for t in batch)
        age_ids, input_ids, posi_ids, segment_ids, attMask, masked_label = batch
        loss, pred, label = model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=attMask,
                                  masked_lm_labels=masked_label)

        if global_params['gradient_accumulation_steps'] > 1:
            loss = loss / global_params['gradient_accumulation_steps']
        loss.backward()

        temp_loss += loss.item()
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if step % 200 == 0:
            print("epoch: {}\t| Loss: {}\t| precision: {}".format(e, temp_loss / 200, cal_acc(label, pred)))
            temp_loss = 0

        if (step + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()
            optim.zero_grad()
print('starting epoch0')


for e in range(50):
    train(model, e, trainload, optim)

    print("** ** * Saving fine - tuned model ** ** * ")
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(global_params['output_dir'], "BEHRT_MLM_DMAY_4graphtests.bin")
    #         create_folder(global_params['output_dir'])
    print('done epoch', e)

    if global_params['save_model']:
        torch.save(model_to_save.state_dict(), output_model_file)