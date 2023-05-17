
from torch.utils.data.dataset import Dataset
from ModelPkg.utils import *


class RPLoader(Dataset):
    def __init__(self, token2idx, dataframe, max_len, max_age=110, year=False, age_symbol=None, min_visit=5,
                 yearOn=False, year2idx=None):
        # dataframe preproecssing
        # filter out the patient with number of visits less than min_visit
        self.vocab = token2idx
        self.max_len = max_len
        self.code = dataframe.code
        self.age = dataframe.age
        self.label = dataframe.label
        self.age2idx, _ = age_vocab(max_age, year, symbol=age_symbol)
        # self.patindex = dataframe.realI
        # self.gender = dataframe.gender
        # self.region = dataframe.region
        self.yearOn = yearOn
        self.year = dataframe.year
        self.year2idx = year2idx

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """
        # cut data
        age = self.age[index]
        code = self.code[index]
        label = self.label[index]
        # gender = int(self.gender[index])
        # region = int(self.region[index])
        # extract data
        age = age[(-self.max_len + 1):]
        code = code[(-self.max_len + 1):]

        if self.yearOn:
            year = self.year[index][(-self.max_len + 1):]
            if code[0] != 'SEP':
                year = np.append(np.array(year[0]), year)

            year = seq_padding(year, self.max_len, token2idx=self.year2idx)
        else:
            year = [0]
        # avoid data cut with first element to be 'SEP'
        if code[0] != 'SEP':
            code = np.append(np.array(['CLS']), code)
            age = np.append(np.array(age[0]), age)
        else:
            code[0] = 'CLS'

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad age sequence and code sequence
        age = seq_padding(age, self.max_len, token2idx=self.age2idx)

        tokens, code = code2index(code, self.vocab)
        #         _, label = code2index(label, self.vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])
        #         label = seq_padding(label, self.max_len, symbol=-1)

        return torch.LongTensor(age),torch.LongTensor(year), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.FloatTensor([label])
    
        # return torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
        #        torch.LongTensor(mask), torch.LongTensor([region]), torch.LongTensor([gender]), torch.FloatTensor(
        #     [label]), torch.LongTensor([self.patindex[index]]), torch.LongTensor(year)

    def __len__(self):
        return len(self.code)


class RPLoaderKG(Dataset):
    def __init__(self, token2idx, dataframe, max_len, max_age=110, year=False, age_symbol=None, min_visit=5,
                 yearOn=False, year2idx=None,KG=None):
        # dataframe preproecssing
        # filter out the patient with number of visits less than min_visit
        self.vocab = token2idx
        self.max_len = max_len
        self.code = dataframe.code
        self.age = dataframe.age
        self.label = dataframe.label
        self.age2idx, _ = age_vocab(max_age, year, symbol=age_symbol)
        self.patindex = dataframe.realI
        self.gender = dataframe.gender
        self.region = dataframe.region
        self.yearOn = yearOn
        self.year = dataframe.year
        self.year2idx = year2idx
        self.KG = KG
    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """
        # cut data
        age = self.age[index]
        code = self.code[index]
        label = self.label[index]
        gender = int(self.gender[index])
        region = int(self.region[index])
        # extract data
        age = age[(-self.max_len + 1):]
        code = code[(-self.max_len + 1):]

        if self.yearOn:
            year = self.year[index][(-self.max_len + 1):]
            if code[0] != 'SEP':
                year = np.append(np.array(year[0]), year)

            year = seq_padding(year, self.max_len, token2idx=self.year2idx)
        else:
            year = [0]
        # avoid data cut with first element to be 'SEP'
        if code[0] != 'SEP':
            code = np.append(np.array(['CLS']), code)
            age = np.append(np.array(age[0]), age)
        else:
            code[0] = 'CLS'

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad age sequence and code sequence
        age = seq_padding(age, self.max_len, token2idx=self.age2idx)

        tokens, code = code2index(code, self.vocab)
        #         _, label = code2index(label, self.vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])
        #         label = seq_padding(label, self.max_len, symbol=-1)
        KGout = []
        for codeelement in code:
            KGout.append(self.KG[codeelement])
        KGout = np.concatenate(KGout)

        return torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor([region]), torch.LongTensor([gender]), torch.FloatTensor(
            [label]), torch.LongTensor([self.patindex[index]]), torch.LongTensor(year), torch.LongTensor(KGout).reshape(-1)

    def __len__(self):
        return len(self.code)

class MLMLoader(Dataset):
    def __init__(self, token2idx, dataframe, max_len, max_age=110, year=False, age_symbol=None, year2idx=None):
        self.vocab = token2idx
        self.max_len = max_len
        self.code = dataframe.code
        self.age = dataframe.age
        self.age2idx, _ = age_vocab(max_age, year, symbol=age_symbol)
        self.year = dataframe.year
        self.year2idx = year2idx

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """

        # extract data
        age = self.age[index][(-self.max_len+1):]
        code = self.code[index][(-self.max_len+1):]
        year = self.year[index][(-self.max_len+1):]

        # avoid data cut with first element to be 'SEP'
        if code[0] != 'SEP':
            code = np.append(np.array(['CLS']), code)
            age = np.append(np.array(age[0]), age)
            year = np.append(np.array(year[0]), year)

        else:
            code[0] = 'CLS'

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad age sequence and code sequence
        age = seq_padding(age, self.max_len, token2idx=self.age2idx)
        year = seq_padding(year, self.max_len, token2idx=self.year2idx)

        tokens, code, label = random_mask(code, self.vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])
        label = seq_padding(label, self.max_len, symbol=-1)

        return torch.LongTensor(age),torch.LongTensor(year), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), torch.LongTensor(mask), torch.LongTensor(label)

    def __len__(self):
        return len(self.code)



class MLMLoaderKG(Dataset):
    def __init__(self, token2idx, dataframe, max_len, max_age=110, year=False, age_symbol=None,KG=None):
        self.vocab = token2idx
        self.max_len = max_len
        self.code = dataframe.code
        self.age = dataframe.age
        self.age2idx, _ = age_vocab(max_age, year, symbol=age_symbol)
        self.KG = KG
    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """

        # extract data
        age = self.age[index][(-self.max_len+1):]
        code = self.code[index][(-self.max_len+1):]

        # avoid data cut with first element to be 'SEP'
        if code[0] != 'SEP':
            code = np.append(np.array(['CLS']), code)
            age = np.append(np.array(age[0]), age)
        else:
            code[0] = 'CLS'

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad age sequence and code sequence
        age = seq_padding(age, self.max_len, token2idx=self.age2idx)

        tokens, code, label ,coderaw = random_mask_extra(code, self.vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])
        label = seq_padding(label, self.max_len, symbol=-1)
        coderaw = seq_padding(coderaw, self.max_len, symbol=self.vocab['PAD'])


        KGout = []
        for codeelement in coderaw:
            KGout.append(self.KG[codeelement])
        KGout = np.concatenate(KGout)

        return torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor(label), torch.LongTensor(KGout).reshape(-1)

    def __len__(self):
        return len(self.code)



class KG_asSentence_MLM(Dataset):
    def __init__(self, token2idx, dataframe, max_len, max_age=110, year=False, age_symbol=None,KG=None, pureICD = None,maxModality = 10, maskToken=None):
        self.vocab = token2idx
        self.max_len = max_len
        self.maxTok =int (self.max_len/maxModality)

        self.code = dataframe.code
        self.age = dataframe.age
        self.age2idx, _ = age_vocab(max_age, year, symbol=age_symbol)
        self.KG = KG
        self.pureICD = pureICD
        self.maxModality = maxModality
        self.maskTokenKG = maskToken
    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """

        # extract data



        age = np.array( self.age[index])
        code = np.array(self.code[index])

        indx = np.array([i for i,x in enumerate(code) if x in self.pureICD])
        # placeholder for no diags below
        if len(indx)==0:
            code = ['MASK']
            age = ['400']
        else:
            code = code[indx]
            age = age[indx]
        trulen = len(age)

        if trulen < self.maxTok:

            age = age[(-self.maxTok + 1):]
            code = code[(-self.maxTok + 1):]

        else:
            randpick = np.random.randint(0, trulen - self.maxTok + 1)
            age = age[randpick: randpick + self.maxTok]
            code =code[randpick: randpick + self.maxTok]


        ageOut = []
        for i , x in enumerate(code):
            temp = [age[i]]*(self.maxModality )
            ageOut.append(temp)
        age = np.array(ageOut).flatten()[(-self.max_len+1):]

        # kg pure token input here
        KGout = []
        for codeelement in code:
            tempel = self.KG[codeelement][:self.maxModality]
            KGout.append(tempel)
        code = np.concatenate(KGout).flatten()[(-self.max_len+1):]




        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        mask[np.where(code==0)]=0

        # pad age sequence and code sequence
        age = seq_padding(age, self.max_len, token2idx=self.age2idx)

        tokens, code, label ,coderaw = KG_sequenceMasking(code, self.maskTokenKG)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx_viaAge(age)
        segment = index_seg_viaAge(age)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=0)
        label = seq_padding(label, self.max_len, symbol=-1)




        return torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor(label)

    def __len__(self):
        return len(self.code)





class MLM_KG_EHR_Seq(Dataset):
    def __init__(self, token2idx, dataframe, max_len, max_age=110, year=False, age_symbol=None, KG=None, pureICD = None,maxModality = 10, maskToken=None, breakpt = 200):
        self.vocab = token2idx
        self.breakpt = breakpt

        self.max_len = max_len


        self.EHRmaxlen = self.breakpt
        self.KGmaxlen = self.max_len- self.breakpt

        self.code = dataframe.code
        self.age = dataframe.age
        self.age2idx, _ = age_vocab(max_age, year, symbol=age_symbol)
        self.KG = KG
        self.pureICD = pureICD
        self.maxModality = maxModality
        self.maskTokenKG = maskToken
        self.maxTok =int (self.KGmaxlen/maxModality)

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """

        # extract data
        age = self.age[index][(-self.EHRmaxlen+1):]
        code = self.code[index][(-self.EHRmaxlen+1):]

        # avoid data cut with first element to be 'SEP'
        if code[0] != 'SEP':
            code = np.append(np.array(['CLS']), code)
            age = np.append(np.array(age[0]), age)
        else:
            code[0] = 'CLS'

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.EHRmaxlen)
        mask[len(code):] = 0
        pureCode = code
        pureAge = age

        # pad age sequence and code sequence
        age = seq_padding(age, self.EHRmaxlen, token2idx=self.age2idx)

        tokens, code, label= random_mask(code, self.vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.EHRmaxlen)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        # pad code and label
        code = seq_padding(code, self.EHRmaxlen, symbol=self.vocab['PAD'])
        label = seq_padding(label, self.EHRmaxlen, symbol=-1)


        age2, code2, position2, segment2, mask2, label2 = self.getKG(pureCode, pureAge, position, segment)

        age, code, position, segment, mask, label = torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor(label)

        age = torch.cat((age, age2))
        code = torch.cat((code, code2))

        position = torch.cat((position, position2))

        segment = torch.cat((segment, segment2))

        mask = torch.cat((mask, mask2))
        label = torch.cat((label, label2))

        return age, code, position, segment, mask ,label

    def __len__(self):
        return len(self.code)

    def getKG(self, code, age, position, segment):


        age= np.array(age)
        code = np.array(code)
        position = np.array(position)
        segment = np.array(segment)

        indx = np.array([i for i,x in enumerate(code) if x in self.pureICD])
        # placeholder for no diags below

        if len(indx)==0:
            code = ['PAD']
            age = ['400']
            segment = segment[(-self.maxTok + 1):]
            position = position[(-self.maxTok + 1):]
        else:
            position = position[indx]
            ageOut = []
            codeOut =[]
            positionOut=[]
            segmentOut =[]
            pos0=position[0]

            for i, ell in enumerate(position):
                if pos0==ell:
                    ageOut.append(age[indx[i]])
                    codeOut.append(code[indx[i]])
                    positionOut.append(position[i])
                    segmentOut.append(segment[indx[i]])
                else:
                    ageOut.append(age[indx[i-1]])
                    codeOut.append('SEP')
                    positionOut.append(position[i-1])
                    segmentOut.append(segment[indx[i-1]])
                    pos0=ell
                    ageOut.append(age[indx[i]])
                    codeOut.append(code[indx[i]])
                    positionOut.append(position[i])
                    segmentOut.append(segment[indx[i]])



            age = ageOut[(-self.maxTok+1):]
            code =codeOut[(-self.maxTok+1):]
            segment = segmentOut[(-self.maxTok+1):]
            position =positionOut[(-self.maxTok+1):]



        ageOut = []
        posOut = []
        segOut=[]
        for i , x in enumerate(code):

            if x!='SEP':
                temp = [age[i]]*(self.maxModality )
                ageOut.append(temp)

                temp = [position[i]]*(self.maxModality )
                posOut.append(temp)

                temp = [segment[i]]*(self.maxModality )
                segOut.append(temp)
            else:
                temp = [age[i]]
                ageOut.append(temp)

                temp = [position[i]]
                posOut.append(temp)

                temp = [segment[i]]
                segOut.append(temp)


        age = np.hstack(np.array(ageOut))[(-self.KGmaxlen+1):]

        position =  np.hstack(np.array(posOut))[(-self.KGmaxlen+1):]
        segment = np.hstack(np.array(segOut))[(-self.KGmaxlen+1):]

        # kg pure token input here
        KGout = []
        rawElems=[]
        for codeelement in code:
            if codeelement!='SEP':
                tempel = self.KG[codeelement][:self.maxModality]
                KGout.append(tempel)
                rawElems.append(tempel)
            else:
                KGout.append([self.maskTokenKG-1])
                rawElems.append([-1])

        rawElems =  np.hstack(np.array(rawElems))[(-self.KGmaxlen+1):]

        code = np.concatenate(KGout).flatten()[(-self.KGmaxlen+1):]




        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.KGmaxlen)
        mask[len(code):] = 0

        mask[np.where(rawElems==0)]=0

        # pad age sequence and code sequence
        age = seq_padding(age, self.KGmaxlen, token2idx=self.age2idx)

        tokens, code, label ,coderaw = KG_sequenceMasking(code, self.maskTokenKG, badTokens =[0, self.maskTokenKG-1])

        # get position code and segment code
        # tokens = seq_padding(tokens, self.KGmaxlen)
        # position = position_idx(tokens)
        # segment = index_seg(tokens)

        # pad code and label
        code = seq_padding(code, self.KGmaxlen, symbol=0)
        label = seq_padding(label, self.KGmaxlen, symbol=-1)
        position = seq_padding(position, self.KGmaxlen, symbol=position[-1])
        segment = seq_padding(segment, self.KGmaxlen, symbol=segment[-1])

        return torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor(label)


class MLM_KG_EHR_Seqv2(Dataset):
    def __init__(self, token2idx, dataframe, max_len, max_age=110, year=False, age_symbol=None, KG=None, pureICD = None,maxModality = 10, maskToken=None, breakpt = 200):
        self.vocab = token2idx
        self.breakpt = breakpt

        self.max_len = max_len


        self.EHRmaxlen = self.breakpt
        self.KGmaxlen = self.max_len- self.breakpt

        self.code = dataframe.code
        self.age = dataframe.age
        self.age2idx, _ = age_vocab(max_age, year, symbol=age_symbol)
        self.KG = KG
        self.pureICD = pureICD
        self.maxModality = maxModality
        self.maskTokenKG = maskToken
        self.maxTok =int (self.KGmaxlen/maxModality)

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """

        # extract data
        age = self.age[index][(-self.EHRmaxlen+1):]
        code = self.code[index][(-self.EHRmaxlen+1):]

        # avoid data cut with first element to be 'SEP'
        if code[0] != 'SEP':
            code = np.append(np.array(['CLS']), code)
            age = np.append(np.array(age[0]), age)
        else:
            code[0] = 'CLS'

        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.EHRmaxlen)
        mask[len(code):] = 0
        pureCode = code
        pureAge = age

        # pad age sequence and code sequence
        age = seq_padding(age, self.EHRmaxlen, token2idx=self.age2idx)

        tokens, code, label= random_mask(code, self.vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.EHRmaxlen)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        # pad code and label
        code = seq_padding(code, self.EHRmaxlen, symbol=self.vocab['PAD'])
        label = seq_padding(label, self.EHRmaxlen, symbol=-1)


        age2, code2, position2, segment2, mask2, label2 = self.getKG(pureCode, pureAge, position, segment)

        age, code, position, segment, mask, label = torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor(label)

        age = torch.cat((age, age2))
        code = torch.cat((code, code2))

        position = torch.cat((position, position2))

        segment = torch.cat((segment, segment2))

        mask = torch.cat((mask, mask2))
        label = torch.cat((label, label2))

        return age, code, position, segment, mask ,label

    def __len__(self):
        return len(self.code)

    def getKG(self, code, age, position, segment):


        age= np.array(age)
        code = np.array(code)
        position = np.array(position)
        segment = np.array(segment)

        indx = np.array([i for i,x in enumerate(code) if x in self.pureICD])
        # placeholder for no diags below

        if len(indx)==0:
            code = ['PAD']
            age = ['400']
            # segment = segment[(-self.maxTok + 1):]
            # position = position[(-self.maxTok + 1):]
        else:
            position = position[indx][(-self.maxTok+1):]
            age = age[indx][(-self.maxTok+1):]
            segment = segment[indx][(-self.maxTok+1):]
            code = code[indx][(-self.maxTok+1):]


        ageOut=[]
        segOut=[]
        posOut=[]
        for i , x in enumerate(code):

            temp = [age[i]]*(self.maxModality )
            ageOut.append(temp)

            temp = [position[i]]*(self.maxModality )
            posOut.append(temp)

            temp = [segment[i]]*(self.maxModality )
            segOut.append(temp)



        age = np.hstack(np.array(ageOut))[(-self.KGmaxlen+1):]

        position =  np.hstack(np.array(posOut))[(-self.KGmaxlen+1):]
        segment = np.hstack(np.array(segOut))[(-self.KGmaxlen+1):]

        # kg pure token input here
        KGout = []
        for codeelement in code:
            tempel = self.KG[codeelement][:self.maxModality]
            KGout.append(tempel)

        code = np.concatenate(KGout).flatten()[(-self.KGmaxlen+1):]



        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.KGmaxlen)
        mask[len(code):] = 0

        mask[np.where(code==0)]=0

        # pad age sequence and code sequence
        age = seq_padding(age, self.KGmaxlen, token2idx=self.age2idx)

        tokens, code, label ,coderaw = KG_sequenceMasking(code, self.maskTokenKG, badTokens =[0])

        # get position code and segment code
        # tokens = seq_padding(tokens, self.KGmaxlen)
        # position = position_idx(tokens)
        # segment = index_seg(tokens)

        # pad code and label
        code = seq_padding(code, self.KGmaxlen, symbol=0)
        label = seq_padding(label, self.KGmaxlen, symbol=-1)
        position = seq_padding(position, self.KGmaxlen, symbol=position[-1])
        segment = seq_padding(segment, self.KGmaxlen, symbol=segment[-1])

        return torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor(label)
