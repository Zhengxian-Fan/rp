import numpy as np
import pandas as pd
import _pickle as pickle
import random
import torch.nn as nn
import torch
import os
import sklearn.metrics as skm



def GPLoad(model, likelihood, filepath, custom = None):
    pre_bert= filepath

    pretrained_dict = torch.load(pre_bert, map_location= 'cpu')
    pretrained_dict = pretrained_dict['model']
    modeld = model.state_dict()
    # 1. filter out unnecessary keys
    if custom==None:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modeld }
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modeld  and k not in custom}

    modeld.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(modeld)

    pre_bert = filepath

    pretrained_dict = torch.load(pre_bert, map_location='cpu')
    likelihood.load_state_dict(pretrained_dict["likelihood"])
    return model, likelihood
def explode(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    fullcols = list(df.columns)
    if (lst_cols is not None
            and len(lst_cols) > 0
            and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
        col: np.repeat(df[col].values, lens)
        for col in idx_cols},
        index=idx)
           .assign(**{col: np.concatenate(df.loc[lens > 0, col].values)
                      for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens == 0, idx_cols], sort=False)
               .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    res = res.reindex(columns=fullcols)
    return res


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def code2index(tokens, token2idx, alt=None):
    if alt is None:
        alt = token2idx['PAD']
    output_tokens = []
    for i, token in enumerate(tokens):
        output_tokens.append(token2idx.get(token, alt))
    return tokens, output_tokens



def random_mask(tokens, token2idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 :
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])
                output_label.append(token2idx.get(token, token2idx['UNK']))

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))
                output_label.append(token2idx.get(token, token2idx['UNK']))

            else:
                output_label.append(-1)
                output_token.append(token2idx.get(token, token2idx['UNK']))
            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def random_mask_extra(tokens, token2idx):
    output_label = []
    output_token = []
    output_tokenraw = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 :
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))
        output_tokenraw.append(token2idx.get(token, token2idx['UNK']))
    return tokens, output_token, output_label,output_tokenraw

def KG_sequenceMasking(tokens, maskTokenKG, badTokens =[0]):
    output_label = []
    output_token = []
    output_tokenraw = []

    for i, token in enumerate(tokens):
        if token not in badTokens:
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15 :
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    output_token.append(maskTokenKG)
                    output_label.append(token)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    output_token.append(random.choice(list(range(1,maskTokenKG))))
                    output_label.append(token)
                else:
                    output_label.append(-1)
                    output_token.append(token)
                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
                output_token.append(token)


        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token)
        output_tokenraw.append(token)

    return tokens, output_token, output_label,output_tokenraw

def random_mask_atleast(tokens, token2idx, limitnum):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 and i >= limitnum:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def random_mask_selective(tokens, token2idx, idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 and i != 0 and i < idx:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def make_weights_for_balanced_classes_CatBased(fulld, nclasses, split):
    count = fulld.DataCat.value_counts()
    print(fulld.DataCat.value_counts())
    weight_per_class = [0.] * nclasses
    #     print(weight_per_class)
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / (float(count[i]))
    weight = [0] * int(N)
    #     print(N)
    #     print(weight_per_class)
    weight_per_class[2] = weight_per_class[2] * split
    weight_per_class[3] = weight_per_class[3] * split

    print(weight_per_class)

    for idx, val in enumerate(fulld.DataCat):
        weight[idx] = weight_per_class[int(val)]
    return weight


def make_weights_for_balanced_classes(fulld, nclasses, split):
    count = fulld.diseaseLabel.value_counts().tolist()

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / (float(count[i]))
    weight = [0] * int(N)
    #     print(weight_per_class)
    weight_per_class[0] = weight_per_class[0] * split
    #     print(weight_per_class)

    for idx, val in enumerate(fulld.diseaseLabel):
        weight[idx] = weight_per_class[int(val)]
    return weight


def make_weights_for_balanced_classes(fulld, nclasses, split, exp=None, out=None):
    if exp is None:
        exp = 'diseaseLabel'
    count = fulld[exp].value_counts().tolist()

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / (float(count[i]))
    weight = [0] * int(N)
    #     print(weight_per_class)
    weight_per_class[0] = weight_per_class[0] * split
    #     print(weight_per_class)

    for idx, val in enumerate(fulld[exp]):
        weight[idx] = weight_per_class[int(val)]
    return weight


def set_random_mask(tokens, token2idx):
    output_label = []
    output_token = []
    if len(tokens) > 5:
        randomI = random.choice(range(len(tokens)))
        for i, token in enumerate(tokens):
            prob = random.random()
            if i == randomI:
                # mask token with 15% probability
                if prob < 0.5:

                    # 80% randomly change token to mask token
                    output_token.append(token2idx["MASK"])
                    output_label.append(token2idx.get(token, token2idx['UNK']))

                    # 10% randomly change token to random token
                elif prob >= 0.5:
                    output_token.append(random.choice(list(token2idx.values())))
                    output_label.append(token2idx.get(token, token2idx['UNK']))
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
                output_token.append(token2idx.get(token, token2idx['UNK']))
        return tokens, output_token, output_label

    else:

        return newrandom_mask(tokens, token2idx)


def newrandom_mask(tokens, token2idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def random_mask_Age(tokens, token2idx):
    output_label = []
    output_token = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def norandom_mask(tokens, token2idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token


def index_seg_viaAge(age):
    seg = []
    flag = 0
    age0= age[0]
    for i ,token in enumerate(age):
        if token != age0:
            seg.append(flag)
            flag =1-flag
            age0=token
        else:
            seg.append(flag)
    return seg
def index_seg(tokens, symbol='SEP'):
    flag = 0
    seg = []

    for token in tokens:
        if token == symbol:
            seg.append(flag)
            if flag == 0:
                flag = 1
            else:
                flag = 0
        else:
            seg.append(flag)
    return seg


def position_idx(tokens, symbol='SEP'):
    pos = []
    flag = 0

    for token in tokens:
        if token == symbol:
            pos.append(flag)
            flag += 1
        else:
            pos.append(flag)
    return pos

def position_idx_viaAge(age):
    pos = []
    flag = 0
    age0= age[0]
    for i ,token in enumerate(age):
        if token != age0:
            pos.append(flag)
            flag += 1
            age0=token
        else:
            pos.append(flag)
    return pos
def age_vocab(max_age, year=False, symbol=None):
    age2idx = {}
    idx2age = {}
    if symbol is None:
        symbol = ['PAD', 'UNK']

    for i in range(len(symbol)):
        age2idx[str(symbol[i])] = i
        idx2age[i] = str(symbol[i])

    if year:
        for i in range(max_age):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)
    else:
        for i in range(max_age * 12):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)

    return age2idx, idx2age

def printCuda(dev=[0,1], multi=False):
    if multi==False:
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print()

        # Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
    else:

        device = torch.device('cuda')
        print()
        print('Using device:', device)

        # Additional Info when using cuda
        for x in dev:
            print('device: ',x)
            if device.type == 'cuda':
                print(torch.cuda.get_device_name(x))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(x) / 1024 ** 3, 1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(x) / 1024 ** 3, 1), 'GB')
def seq_padding(tokens, max_len, token2idx=None, symbol=None):
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                # 1 indicate UNK
                seq.append(token2idx.get(tokens[i], token2idx['UNK']))
            else:
                seq.append(token2idx.get(symbol))
    return seq


def seq_padding_reverse(tokens, max_len, token2idx=None, symbol=None):
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(tokens)
    tokens = tokens[::-1]
    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                # 1 indicate UNK
                seq.append(token2idx.get(tokens[i], token2idx['UNK']))
            else:
                seq.append(token2idx.get(symbol))
    return seq[::-1]


def age_seq_padding(tokens, max_len, token2idx=None, symbol=None):
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                # 1 indicate UNK
                seq.append(token2idx[tokens[i]])
            else:
                seq.append(token2idx[symbol])
    return seq


def BPsplitMaker(start, finish, BPSPLIT, min=0):
    # for the treatment prediction bucket
    splits = np.linspace(start, finish, BPSPLIT)
    splits = np.insert(splits, 0, min)
    splits = np.insert(splits, len(splits), float('Inf'))
    t = {}
    for i, x in enumerate(splits[:-1]):
        t[float(i)] = str(x) + "--" + str(splits[i + 1])
    t = list(t.values())
    return (splits, t)


def cal_acc(label, pred, logS=True):
    logs = nn.LogSoftmax()
    label = label.cpu().numpy()
    ind = np.where(label != -1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    if logS == True:
        truepred = logs(torch.tensor(truepred))
    else:
        truepred = torch.tensor(truepred)
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    precision = skm.precision_score(truelabel, outs, average='micro')

    return precision

def cal_acc(label, pred, logS=True):
    logs = nn.LogSoftmax()
    label = label.cpu().numpy()
    ind = np.where(label != -1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    if logS ==True:
        truepred = logs(torch.tensor(truepred))
    else:
        truepred = torch.tensor(truepred)
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    precision = skm.precision_score(truelabel, outs, average='micro')

    return precision

def partition(values, indices):
    idx = 0
    for index in indices:
        sublist = []
        idxfill = []
        while idx < len(values) and values[idx] <= index:
            # sublist.append(values[idx])
            idxfill.append(idx)

            idx += 1
        if idxfill:
            yield idxfill

def subsampleLiveFunction(x, kgtemp, thres):
    toDel = []
    toKeep = []
    if len(kgtemp[1]) > thres:
        for i, y in enumerate(kgtemp[1]):
            if x not in y:
                toDel.append(y)
            else:
                toKeep.append(y)
        random.shuffle(toDel)
        for delel in toDel[:int((thres-(len(kgtemp[1])-len(toDel)))/2)]:
            toKeep.append(delel)
            toKeep.append((delel[1], delel[0]))
        toKeep = set(toKeep)
        newels = set()
        for el in toKeep:
            newels.add(el[0])
        return [newels, toKeep]


    else:
        return [kgtemp[0], kgtemp[1]]

        #         print(counts0)
        #
        # random.shuffle(toDel)
        # toDel = toDel[:int(((len(kgtemp[1]) - thres)))]
        # newTemp1 = kgtemp[1]
        # #         print(newTemp1)
        # #         print(toDel)
        #
        # for delel in toDel:
        #     if (delel[1], delel[0]) not in toDel:
        #         toDel.append((delel[1], delel[0]))
        # #         print(toDel)
        #
        # for delel in toDel:
        #     newTemp1.remove(delel)
        # #             if (delel[1],delel[0]) not in toDel:
        # #                 newTemp1.remove((delel[1],delel[0]))
        #
        # #         break
        # counts1 = []
        # newels = set()
        # for el in newTemp1:
        #     newels.add(el[0])
        #     counts1.append(el[0])

        # return [newels, newTemp1]

def toLoad(model, filepath, custom=None):
    pre_bert = filepath

    pretrained_dict = torch.load(pre_bert, map_location='cpu')
    modeld = model.state_dict()
    # 1. filter out unnecessary keys
    if custom == None:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modeld}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modeld and k not in custom}
    print(pretrained_dict.keys())
    modeld.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(modeld)
    return model


import sklearn


def OutcomePrecision(logits, label, sig=True):
    sig = nn.Sigmoid()
    if sig == True:
        output = sig(logits)
    else:
        output = logits
    label, output = label.cpu(), output.detach().cpu()
    tempprc = sklearn.metrics.average_precision_score(label.numpy(), output.numpy())
    return tempprc, output, label


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def precision_test(logits, label, sig=True):
    sigm = nn.Sigmoid()
    if sig == True:
        output = sigm(logits)
    else:
        output = logits
    label, output = label.cpu(), output.detach().cpu()

    tempprc = sklearn.metrics.average_precision_score(label.numpy(), output.numpy())
    return tempprc, output, label


def roc_auc(logits, label, sig=True):
    try:
        sigm = nn.Sigmoid()
        if sig == True:
            output = sigm(logits)
        else:
            output = logits
        label, output = label.cpu(), output.detach().cpu()

        tempprc = sklearn.metrics.roc_auc_score(label.numpy(), output.numpy())
        return tempprc, output, label
    except:
        return "error"

def ap(logits, label, sig=True):
    try:
        sigm = nn.Sigmoid()
        if sig == True:
            output = sigm(logits)
        else:
            output = logits
        label, output = label.cpu(), output.detach().cpu()

        tempprc = sklearn.metrics.average_precision_score(label.numpy(), output.numpy())
        return tempprc, output, label
    except:
        return "error"
    
# golobal function
def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)