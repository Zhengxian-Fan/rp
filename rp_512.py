import sys

print('starting run....')

sys.path.insert(0,'/home/zhengxian/BEHRT/ModelPkg/')

from ModelPkg.BEHRTraw import *
import torch.nn as nn
import torch
import numpy as np

from torch.utils.data.dataset import Dataset
import time
from torch.autograd import Variable
from ModelPkg import utils
from ModelPkg.MLMRaw import *

from ModelPkg.DataProc import *
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
from pytorch_pretrained_bert import optimizer
from torch.nn import functional as F

from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

file_config = {
        'vocab': '/home/zhengxian/BEHRT/Data/GeneralVocDMProc_25k',
    'fulld': '/home/zhengxian/flan_t5/data_censored/test.parquet',
    'testd': '/home/zhengxian/flan_t5/data_censored/test.parquet',
    #'fulld': '/gpfs3/well/rahimi/users/gra027/JNb/general_model_newCutCPRD/Data/MLM_for_pretraining_28M_1985_2020__unique_per6m_50pc_sample___10kdebug.parquet/',
    'yearVocab':  '/home/zhengxian/BEHRT/Data/yearVoc_1985_2021',
}


optim_config = {
    'lr': 3e-5,
    'warmup_proportion': 0.1,
}

global_params = {
    'batch_size': 64,
    'gradient_accumulation_steps': 2,
    'device': 'cuda:0',
    'output_dir':'/home/zhengxian/BEHRT/SavedModels/',
    'output_name': 'risk_pred.bin',
    'save_model': True,
    'max_len_seq': 512,
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'min_visit': 1,
    'yearOn':True
}


YearVocab = utils.load_obj(file_config['yearVocab'])
create_folder(global_params['output_dir'])
BertVocab = utils.load_obj(file_config['vocab'])
print(len(BertVocab['token2idx']))

ageVocab, _ = utils.age_vocab(max_age=global_params['max_age'], year=global_params['age_year'], symbol=global_params['age_symbol'])
fulldata = pd.read_parquet(file_config['fulld'])
testdata = pd.read_parquet(file_config['testd'])
print('read data....')

trainSet = RPLoader(token2idx=BertVocab['token2idx'], dataframe=fulldata, max_len=global_params['max_len_seq'], max_age=global_params['max_age'], year=global_params['age_year'], age_symbol=global_params['age_symbol'],year2idx = YearVocab['token2idx'] )
trainload = DataLoader(dataset=trainSet, batch_size=global_params['batch_size'], shuffle=True)

testSet = RPLoader(token2idx=BertVocab['token2idx'], dataframe=testdata, max_len=global_params['max_len_seq'], max_age=global_params['max_age'], year=global_params['age_year'], age_symbol=global_params['age_symbol'],year2idx = YearVocab['token2idx'] )
testload = DataLoader(dataset=testSet, batch_size=global_params['batch_size']*8, shuffle=False)

model_config = {
    'vocab_size': len(BertVocab['token2idx'].keys()), # number of disease + symbols for word embedding
    'hidden_size': 150, # word embedding and seg embedding hidden size
    'seg_vocab_size': 2, # number of vocab for seg embedding
    'age_vocab_size': len(ageVocab.keys()), # number of vocab for age embedding
    'year_vocab_size': len(YearVocab['token2idx'].keys()), # number of vocab for age embedding
    'max_position_embedding': global_params['max_len_seq'], # maximum number of tokens
    'hidden_dropout_prob': 0.1, # dropout rate
    'num_hidden_layers': 6, # number of multi-head attention layers required
    'num_attention_heads': 6, # number of attention heads
    'attention_probs_dropout_prob': 0.1, # multi-head attention dropout rate
    'intermediate_size': 108, # the size of the "intermediate" layer in the transformer encoder
    'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    'initializer_range': 0.02, # parameter weight initializer range,
    'yearOn':True,
    'year_vocab_size': len(YearVocab['token2idx'].keys()),

}
conf = BertConfig(model_config)
model = BertRP(conf, num_labels=1)
output_model_file = os.path.join(global_params['output_dir'], "HF_RP_new_pheno_slen_512_6msummary.bin")
torch.load(output_model_file)
model = toLoad(model, output_model_file, custom=['bert.embeddings.posi_embeddings.weight'])
model = model.to(global_params['device'])
optim = optimizer.adam(params=list(model.named_parameters()), config=optim_config)

BertVocab = utils.load_obj(file_config['vocab'])

def do_eval(dataset, model):
    all_logits = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataset, desc="Evaluating"):
            batch = tuple(t.to(global_params['device']) for t in batch)
            age_ids, year_ids, input_ids, posi_ids, segment_ids, attMask, labels = batch
            loss, logits = model(input_ids, age_ids, segment_ids, posi_ids, year_ids, attention_mask=attMask, labels=labels)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # print(all_logits, all_labels, roc_auc(logits, labels)[0], ap(logits, labels)[0])
    model.train()
    return roc_auc(all_logits, all_labels)[0], ap(all_logits, all_labels)[0]


# df = pd.read_parquet('/home/zhengxian/flan_t5/data_censored/test.parquet')

# indices = [ 5,  30,  37, 126, 160, 168, 196, 269, 289, 406, 407, 409, 446, 450, 480]
# rows = df.iloc[indices].to_csv('wrong.csv')
# print(rows)

# rows = df.iloc[~df.index.isin(indices)][:512].to_csv('correct.csv')

def do_check(dataset, model):
    all_logits = []
    all_labels = []

    false_length = []
    correct_length = []

    false_age = []
    correct_age = []

    false_year = []
    correct_year = []

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataset, desc="Evaluating")):
            batch = tuple(t.to(global_params['device']) for t in batch)
            age_ids, year_ids, input_ids, posi_ids, segment_ids, attMask, labels = batch
            loss, logits = model(input_ids, age_ids, segment_ids, posi_ids, year_ids, attention_mask=attMask, labels=labels)
            # print(loss.size(), logits.size())

            all_probs = torch.sigmoid(logits)
            # print(all_probs)
            predicted_labels = (all_probs >= 0.5).float()
            false_predictions = (predicted_labels != labels)
            false_probs = all_probs[false_predictions.squeeze(dim=1)]
            false_indices = (false_predictions.squeeze(dim=1)).nonzero().squeeze(dim=1)

            last_nonzero_indices = torch.gt(age_ids, 0).sum(dim=1) - 1
            last_nonzero_values = age_ids[torch.arange(age_ids.size(0)), last_nonzero_indices]

            false_length.append(torch.mean(torch.sum(attMask[false_indices], dim=1).float()).reshape(1))
            correct_length.append(torch.mean(torch.sum(attMask[~false_indices], dim=1).float()).reshape(1))
            # print(age_ids[torch.arange(age_ids.size(0)), last_nonzero_indices])
            # print(age_ids[false_indices].size())
            # exit()
            false_age.append(torch.mean(age_ids[torch.arange(age_ids.size(0)), last_nonzero_indices].float()).reshape(1))
            correct_age.append(torch.mean(age_ids[torch.arange(age_ids.size(0)), last_nonzero_indices].float()).reshape(1))

            false_year.append(torch.mean(year_ids[torch.arange(age_ids.size(0)), last_nonzero_indices].float()).reshape(1))
            correct_year.append(torch.mean(year_ids[torch.arange(age_ids.size(0)), last_nonzero_indices].float()).reshape(1))


            # if idx >=2:
            #     break
            # print(false_length, correct_length)
            # all_logits.append(logits.cpu())
            # all_labels.append(labels.cpu())
            # break
    # print(false_length)

    false_length = torch.cat(false_length, dim=0)
    correct_length = torch.cat(correct_length, dim=0)
    print('false length', torch.mean(false_length))
    print('correct length', torch.mean(correct_length))
    print('false age', torch.mean(false_age))
    print('correct age', torch.mean(correct_age))
    print('false year', torch.mean(false_year))
    print('correct year', torch.mean(correct_year))
    # all_probs = torch.sigmoid(all_logits)
    # predicted_labels = (all_probs >= 0.5).float()
    # false_predictions = (predicted_labels != all_labels)
    # false_probs = all_probs[false_predictions.squeeze(dim=1)]
    # false_indices = (false_predictions.squeeze(dim=1)).nonzero().squeeze(dim=1)
    # print(torch.mean(torch.sum(attMask[false_indices], dim=1).float()))
    # print(torch.mean(torch.sum(attMask[~false_indices], dim=1).float()))
    exit()
    probs = torch.sigmoid(all_logits)

    # print(all_logits, all_labels, roc_auc(logits, labels)[0], ap(logits, labels)[0])
    model.train()
    return roc_auc(all_logits, all_labels)[0], ap(all_logits, all_labels)[0]


output_model_file = os.path.join(global_params['output_dir'], "HF_RP_new_pheno_slen_512_6msummary.bin")
best_ap = 0

# auroc_value, ap_value = do_eval(testload, model)
# print("auc-roc, ap", auroc_value, ap_value)

auroc_value, ap_value = do_check(testload, model)
print("auc-roc, ap", auroc_value, ap_value)

exit()

def train(model, e, validload, optim):
    global best_ap
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    pbar = tqdm(validload, desc="Training")

    for step, batch in enumerate(pbar):
        batch = tuple(t.to(global_params['device']) for t in batch)
        # print(batch)
        age_ids, year_ids,input_ids, posi_ids, segment_ids, attMask, labels = batch

        loss, logits = model(input_ids, age_ids, segment_ids, posi_ids,year_ids, attention_mask=attMask,
                                  labels=labels)

        if global_params['gradient_accumulation_steps'] > 1:
            loss = loss / global_params['gradient_accumulation_steps']
        loss.backward()

        temp_loss += loss.item()
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if step % 100 == 0:
            print("epoch: {}\t| Loss: {}\t| auc-roc: {}, ap: {}".format(e, temp_loss / 200, roc_auc(logits, labels)[0], ap(logits, labels)[0]))
            temp_loss = 0

        if (step + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()
            optim.zero_grad()
        if (step+1 )%3000 ==0:
            
            print("mid epoch: " +str(step *148) + "..... ** ** * Saving fine - tuned model ** ** * ")
            auroc_value, ap_value = do_eval(testload, model)
            print("auc-roc, ap", auroc_value, ap_value)
            if ap_value > best_ap:
                torch.save(model_to_save.state_dict(), output_model_file)
                best_ap = ap_value
                
print('starting epoch0')

model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
# output_model_file = os.path.join(global_params['output_dir'], "HF_RP_6msummary_final.bin")

print('to__save: output_model_file')
for e in range(50):
    train(model, e, trainload, optim)

    print("** ** * Saving fine - tuned model ** ** * ")
    torch.save(model_to_save.state_dict(), output_model_file)

    # create_folder(global_params['output_dir'])
    print('done epoch', e)