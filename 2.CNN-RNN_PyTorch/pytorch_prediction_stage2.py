#!/usr/bin/env python
# coding: utf-8

def stage2_predict(VALID, TEST, RNN_Type, SEQ_SIZE, CNN_fix):

    # In[1]:


    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt


    # In[2]:
    

    VALID = VALID[0]
    TEST = TEST[0]
    
    print(f'VALID: {VALID}')
    print(f'TEST: {TEST}')
    
    print(RNN_Type)
    print(SEQ_SIZE)
    print(CNN_fix)


    # In[6]:


    df = pd.read_csv('./data_64_5mice.csv')
    df


    # In[7]:


    def get_filename(mouse_No):
        filename = np.empty(0)
        for i in mouse_No:
            files = df[df['Mouse_#'] == i]['Data_ID'].values
            filename = np.hstack([filename, files])
        return filename


    # In[8]:


    import tifffile
    import gc

    def load_image(filename):
        raw_image = tifffile.imread(filename)
        X_tmp = (raw_image - raw_image.mean(axis=0)) / raw_image.std() # normalize
        # 3 channel image
        ax = X_tmp.shape
        X = np.zeros([ax[0] - 2, 3, ax[1], ax[2]])
        X[:,0,:,:] = X_tmp[:-2, :, :]
        X[:,1,:,:] = X_tmp[1:-1, :, :]
        X[:,2,:,:] = X_tmp[2:, :, :]
        del X_tmp
        gc.collect();
        return X


    # In[9]:


    from scipy import io

    def load_runrest(filename):
        content = io.loadmat(filename)
        Y = content['runrest'][:,1:-1].reshape([-1,1])
        del content
        gc.collect();
        return Y


    #  # Stage1

    # In[10]:


    import torch
    from torch import nn, optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader


    # In[11]:


    BATCH_SIZE = 8 # 64
    VER = 'b0'
    from efficientnet_pytorch import EfficientNet
    cnn = EfficientNet.from_pretrained('efficientnet-'+VER, num_classes=1)


    # In[12]:


    FEATURE_SIZE = 1280 # efficient-net b0の時は1280
    RNN_UNITS = 128

    if RNN_Type == 'RNN':
        rnn = nn.RNN(FEATURE_SIZE, RNN_UNITS, num_layers=2, dropout=0.2, batch_first=True)
    elif RNN_Type == 'LSTM':
        rnn = nn.LSTM(FEATURE_SIZE, RNN_UNITS, num_layers=2, dropout=0.2, batch_first=True)
    elif RNN_Type == 'GRU':
        rnn = nn.GRU(FEATURE_SIZE, RNN_UNITS, num_layers=2, dropout=0.2, batch_first=True)


    # In[13]:


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


    # In[14]:


    # Model
    model = CNN_RNN(cnn).cuda()


    # In[15]:


    # Load model
    model.load_state_dict(torch.load(f'./stage2_model/model_EfficientNet{VER}_{RNN_Type}_seq{SEQ_SIZE}_valid{VALID}_test{TEST}_{CNN_fix}.pth'))


    # In[16]:


    # For prediction
    model.eval()


    # In[17]:


    class ImageDataset:
        def __init__(self, features, targets):
            self.features = features
            self.targets = targets

        def __len__(self):
            return (self.targets.shape[0] - 2*SEQ_SIZE)

        def __getitem__(self, idx):
            x = torch.tensor(self.features[idx:idx+2*SEQ_SIZE+1, :, :, :], dtype=torch.float)
            y = torch.tensor(self.targets[idx+SEQ_SIZE, :], dtype=torch.float)            
            return x, y


    # In[18]:

    image_path = './data_of_image_64_5mice/'
    behavior_path = './data_of_behavior_64_5mice/'

    def inference_1_image(filename):
        X = load_image(image_path + filename + '.tif')
        Y = load_runrest(behavior_path + filename + '_behavior.mat')
        ds_i = ImageDataset(X, Y)
        Y_pred = np.zeros([len(ds_i), 1])
        Y_true = np.zeros([len(ds_i), 1])
        infe = DataLoader(ds_i, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, pin_memory=True)
        for idx, (imgs,labels) in enumerate(infe):
            bs = labels.shape[0]
            Y_pred[idx*BATCH_SIZE:idx*BATCH_SIZE+bs] = torch.sigmoid(model(imgs.cuda())).cpu().detach().numpy()
            Y_true[idx*BATCH_SIZE:idx*BATCH_SIZE+bs] = labels
        del X, Y, ds_i, infe
        gc.collect();
        return Y_pred, Y_true


    # In[19]:


    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc, roc_auc_score

    def calculate_accuracy(filemanes):
        Y_pred_all = []
        Y_true_all = []
        acc = np.zeros(len(filemanes))
        for i, filename in enumerate(filemanes):
            print(filename)
            Y_pred, Y_true = inference_1_image(filename)
            Y_pred_all.append(Y_pred)
            Y_true_all.append(Y_true)
            output = Y_pred > 0.5
            accuracy = (output == Y_true).mean()
            print('Accuracy: {}'.format(accuracy))
            AUC = roc_auc_score(Y_true, Y_pred)
            print(f'AUC: {AUC}')
            acc[i] = accuracy
        print('Mean Accuracy: {}'.format(acc.mean()))
        return Y_pred_all, Y_true_all


    # In[20]:


    infer_data = list(get_filename([TEST]))
    infer_data


    # In[21]:


    Y_pred, Y_true = calculate_accuracy(infer_data)


    # In[22]:


    # Save test result
    from scipy import io
    io.savemat(f"./result/test_result_EfficientNet{VER}_{RNN_Type}_seq{SEQ_SIZE}_valid{VALID}_test{TEST}_{CNN_fix}.mat", 
               {"test_label":Y_true, "test_pred":Y_pred})


    # In[23]:


    # Plot ROC curve
    def plot_roc(y_pred, y_test, mode):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        AUC = roc_auc_score(y_test, y_pred)
        plt.plot(fpr, tpr, label="AUC = {:.3f}".format(AUC))
        plt.title(mode + " ROC curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()
        print('AUC: {:.3f}'.format(AUC))
        opt_thr = thresholds[np.argmax(tpr - fpr)]
        print('Optimal threshold: {:.3f}'.format(opt_thr))


    # In[24]:


    test_pred = np.vstack(Y_pred)
    test_true = np.vstack(Y_true)


    # In[25]:


    plot_roc(test_pred, test_true, "TEST")


    # In[ ]:




