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

# # EfficientNetB0 Pretraining


def stage1_train(TRAIN, VALID, TEST):
    # In[1]:

    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt


    # In[3]:


    TRAIN = f'[{TRAIN[0]}{TRAIN[1]}{TRAIN[2]}]'
    VALID = VALID[0]
    TEST = TEST[0]


    # In[4]:


    print(TRAIN)
    print(VALID)
    print(TEST)


    # # Training

    # In[5]:


    import torch
    from torch import nn, optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader


    # In[6]:


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


    # In[7]:


    BATCH_SIZE = 512
    VER = 'b0'
    from efficientnet_pytorch import EfficientNet
    cnn = EfficientNet.from_pretrained('efficientnet-'+VER, num_classes=1)


    # In[8]:


    # Model
    model = cnn.cuda()


    # In[9]:


    # Count number of parameters
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(params) 


    # In[10]:


    df = pd.read_csv('./data_64_5mice.csv')
    df


    # In[11]:


    # parameter fof normalizing
    mean_dict = {}
    std_dict = {}
    save_path = './mean_and_std/'

    for i in df['Data_ID']:
        m = np.load(save_path + i + '_mean.npy')
        s = np.load(save_path + i + '_std.npy')
        mean_dict[i] = m
        std_dict[i] = s


    # In[12]:


    import glob
    import re


    # In[13]:


    # Dataset
    class ImageLoadDataset:
        def __init__(self, file_names):
            self.files = file_names

        def __len__(self):
            return (len(self.files))

        def __getitem__(self, idx):
            npz_path = self.files[idx]
            xy_data = np.load(npz_path)
            experiment_name = npz_path[len('./npz_data/1/'):-len(re.search('_\d+.npz',npz_path).group())] # Data_ID
            mean = mean_dict[experiment_name]
            std = std_dict[experiment_name]
            x_tmp = (xy_data['image'] - mean) / std # Normalize
            x = torch.tensor(x_tmp, dtype=torch.float)
            y = torch.tensor(xy_data['label'], dtype=torch.float)            
            return x, y


    # In[14]:


    # Dataset setting
    from fastai.data.core import DataLoaders
    ds_train = ImageLoadDataset(glob.glob(f'./npz_data/{TRAIN}/*.npz'))
    ds_valid = ImageLoadDataset(glob.glob(f'./npz_data/{VALID}/*.npz'))
    dls = DataLoaders.from_dsets(ds_train, ds_valid, bs=BATCH_SIZE, shuffle=True,
            num_workers=10, pin_memory=True, worker_init_fn=seed_worker, generator=generator).cuda()


    # In[15]:


    # Learn setting
    from fastai.vision.all import Learner, ShowGraphCallback, SaveModelCallback
    loss = nn.BCEWithLogitsLoss()
    learn = Learner(dls, model, loss_func=loss)
    callbacks = [SaveModelCallback(monitor='valid_loss', comp=np.less), ShowGraphCallback()]


    # In[16]:


    # Training
    Epoch = 3
    learn.fit_one_cycle(Epoch, lr_max=1e-3, cbs=callbacks)


    # In[17]:


    # Save model
    torch.save(learn.model.state_dict(),f'./stage1_model/model_EfficientNet{VER}_valid{VALID}_test{TEST}.pth')





