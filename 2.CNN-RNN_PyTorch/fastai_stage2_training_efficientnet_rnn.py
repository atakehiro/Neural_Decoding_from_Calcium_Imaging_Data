#!/usr/bin/env python
# coding: utf-8

# Copyright 2021 Takehiro Ajioka

# ### Enviroment
# 
# Python 3.6
# Anaconda
# pytorch==1.9.0_cuda11.1
# fastai==2.5.2
# tifffile==2020.9.3
# opencv-python==4.5.3.56
# efficientnet-pytorch==0.7.1
# tqdm==4.47.0

# # EfficientNetB0 ＋ GRU

def stage2_train(TRAIN, VALID, TEST, RNN_Type, CNN_fix, SEQ_SIZE):


    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt


    # In[4]:


    TRAIN = f'[{TRAIN[0]}{TRAIN[1]}{TRAIN[2]}]'
    VALID = VALID[0]
    TEST = TEST[0]


    # In[5]:


    print(TRAIN)
    print(VALID)
    print(TEST)


    # # Training

    # ## Model

    # In[6]:


    import torch
    from torch import nn, optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader


    # In[7]:


    # Random seed
    import os, random
    SEED = 2021

    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms = True
        torch.backends.cudnn.deterministic = True

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    seed_everything(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)


    # In[8]:


    VER = 'b0'
    from efficientnet_pytorch import EfficientNet
    cnn = EfficientNet.from_pretrained('efficientnet-'+VER, num_classes=1)


    # In[9]:


    cnn.load_state_dict(torch.load(f'./stage1_model/model_EfficientNet{VER}_valid{VALID}_test{TEST}.pth'))


    # In[10]:


    FEATURE_SIZE = 1280 # efficient-net b0の時は1280
    RNN_UNITS = 128

    if RNN_Type == 'RNN':
        rnn = nn.RNN(FEATURE_SIZE, RNN_UNITS, num_layers=2, dropout=0.2, batch_first=True)
    elif RNN_Type == 'LSTM':
        rnn = nn.LSTM(FEATURE_SIZE, RNN_UNITS, num_layers=2, dropout=0.2, batch_first=True)
    elif RNN_Type == 'GRU':
        rnn = nn.GRU(FEATURE_SIZE, RNN_UNITS, num_layers=2, dropout=0.2, batch_first=True)


    # In[11]:


    # Model
    class CNN_RNN(nn.Module):
        def __init__(self,cnn):
            super(CNN_RNN, self).__init__()
            self.cnn = cnn
            self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            self.rnn = rnn
            self.linear_out = nn.Linear(RNN_UNITS, 1)

        def forward(self, x):
            x = x.reshape(-1, 3, x.shape[3], x.shape[4]) # (batch, channel, size1, size2)
            x = self.cnn.extract_features(x)
            x = self.avgpool(x)
            x = x.reshape(-1, 2*SEQ_SIZE+1, FEATURE_SIZE) # (batch, seq_len, input_size)
            self.rnn.flatten_parameters()
            o_rnn, _ = self.rnn(x)
            output = self.linear_out(o_rnn[:,-1,:])
            return output


    # In[12]:


    model = CNN_RNN(cnn).cuda()


    # In[13]:


    # Fix the parameters of CNN
    if CNN_fix == 'Fix':
        for param in cnn.parameters():
            param.requires_grad = False
            
    # Batch size
    BATCH_SIZE = 32 #64
    print(f'Batch size: {BATCH_SIZE}')


    # In[14]:


    # Count number of parameters
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(params) 


    # ## Learning

    # In[15]:


    df = pd.read_csv('./data_64_5mice.csv')
    df


    # In[16]:


    # parameter fof normalizing
    mean_dict = {}
    std_dict = {}
    save_path = './mean_and_std/'

    for i in df['Data_ID']:
        m = np.load(save_path + i + '_mean.npy')
        s = np.load(save_path + i + '_std.npy')
        mean_dict[i] = m
        std_dict[i] = s


    # In[17]:


    import glob
    import re


    # In[18]:


    # exclude frame number
    def calc_exc_num(i):
        f_list = glob.glob(i + '*.npz')
        L = len(f_list)
        #print(f'frame number: {L}')
        exc_num = list(range(SEQ_SIZE)) + list(range(L-SEQ_SIZE,L))
        return exc_num


    # In[19]:


    # input file list
    def input_list(MODE):
        folder = glob.glob(f'./npz_data/{MODE}/*_0.npz')
        exp_path = [folder[i].rstrip('_0.npz') for i in range(len(folder))]
        exclude_list = []
        for i in exp_path:
            exc_num = calc_exc_num(i)
            for n in exc_num:
                npz_path = i + '_' + str(n) + '.npz'
                exclude_list.append(npz_path)
        f_list = glob.glob(f'./npz_data/{MODE}/*.npz')
        input_list = [i for i in f_list if i not in exclude_list]
        return input_list


    # In[20]:


    # Dataset
    class ImageLoadDataset:
        def __init__(self, filelist):
            self.files = filelist

        def __len__(self):
            return (len(self.files))

        def __getitem__(self, idx):
            npz_path = self.files[idx]
            xy_data = np.load(npz_path)
            experiment_name = npz_path[len('./npz_data/1/'):-len(re.search('_\d+.npz',npz_path).group())]
            mean = mean_dict[experiment_name]
            std = std_dict[experiment_name]
            ax = xy_data['image'].shape
            num = re.search('_\d+.npz', npz_path).group().lstrip('_').rstrip('.npz') # frame number
            X = np.zeros([2*SEQ_SIZE+1, ax[0], ax[1], ax[2]])
            for idx, i in enumerate(range(int(num) - SEQ_SIZE, int(num) + SEQ_SIZE + 1)):
                length = len(num) + len('_.npz')
                path = npz_path[:-length] + '_' + str(i) + '.npz'
                xy_data2 = np.load(path)
                X[idx,:,:,:] = (xy_data2['image'] - mean) / std # Normalize
            x = torch.tensor(X, dtype=torch.float)
            y = torch.tensor(xy_data['label'], dtype=torch.float)      
            return x, y


    # In[21]:


    # Dataset setting
    from fastai.data.core import DataLoaders
    ds_train = ImageLoadDataset(input_list(TRAIN))
    ds_valid = ImageLoadDataset(input_list(VALID))
    dls = DataLoaders.from_dsets(ds_train, ds_valid, bs=BATCH_SIZE, shuffle=True,
            num_workers=10, pin_memory=True, worker_init_fn=seed_worker, generator=generator).cuda()


    # In[22]:


    # Learn setting
    from fastai.vision.all import Learner, ShowGraphCallback, SaveModelCallback
    loss = nn.BCEWithLogitsLoss()
    learn = Learner(dls, model, loss_func=loss).to_fp16() # Mixed precision
    callbacks = [SaveModelCallback(monitor='valid_loss', comp=np.less), ShowGraphCallback()]


    # In[23]:


    # Training
    Epoch = 3
    learn.fit_one_cycle(Epoch, lr_max=1e-3, cbs=callbacks)


    # In[ ]:


    # Save model
    torch.save(learn.model.state_dict(),f'./stage2_model/model_EfficientNet{VER}_{RNN_Type}_seq{SEQ_SIZE}_valid{VALID}_test{TEST}_{CNN_fix}.pth')




