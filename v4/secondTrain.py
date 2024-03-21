# 每次要检查是否修改 SMALL_MAG 和 CHECK_POINT
# %%
from os import makedirs, path
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import net

# %%
from const import SAMPLES, SMALL_MAG, DEVICE, TORCH_SEED, BATCH_SIZE, NUM_WORKERS, ALPHA, EPOCHES, CHECK_POINT, timeSince
# %%
class secondCitationSet(Dataset):
    def __init__(self, samples: pd.DataFrame, smallMag:pd.DataFrame, transform=None):
        self.samples = samples
        self.smallMag = smallMag
        self.sortedSmallMag = smallMag[['paperID','neighbors']].set_index(['paperID']).sort_index()
        self.transform = transform

        self.smallMagTitleLength = len(smallMag)
        self.length = len(samples)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):      
        fakeTitle, fakeAbstract = self.smallMag.loc[random.randrange(self.smallMagTitleLength), ['titleEmb', 'abstractEmb']]

        _sample = self.samples.iat[idx,0]
        if len(_sample['titles']) > 1:
            n, n2 = random.sample(range(len(_sample['titles'])), k=2)

            title = _sample['titles'][n] # cited title
            abstract = _sample['abstracts'][n] # cited abstract
            paperID = _sample['paperIDs'][n] # cited paperID

            title2 = _sample['titles'][n2] # cited title
            abstract2 = _sample['abstracts'][n2] # cited abstract
        else:
            n = 0
            n2 = 1
            title = _sample['titles'][n] # cited title
            abstract = _sample['abstracts'][n] # cited abstract
            paperID = _sample['paperIDs'][n] # cited paperID 
            title2 = fakeTitle
            abstract2 = fakeAbstract

        neighbors = self.sortedSmallMag.loc[paperID, 'neighbors'][1:]
        neighborIdx = random.choice(neighbors)
        neighborTitle, neighborAbstract = self.smallMag.loc[neighborIdx, ['titleEmb', 'abstractEmb']]

        sample = {'contextRaw': _sample['contextRaw'], 'context': _sample['context'], 'clusterID': n, 'title': title, 'abstract': abstract,'nTitle':neighborTitle, 'nAbstract':neighborAbstract, 'clusterID2': n2, 'title2': title2, 'abstract2': abstract2, 'fTitle': fakeTitle, 'fAbstract': fakeAbstract, 'paperIDRaw': paperID}

        if self.transform:
            sample = self.transform(sample)

        return sample
# %%
def secondTrain(model: nn.Module, train_loader, optimizer, device = DEVICE):
    model.train()
    epoch_loss = 0
    num = 0
    length = 0
    for batch in train_loader:
        for key in batch.keys():
            if 'Raw' not in key:
                batch[key] = batch[key].to(device)
        context = batch['context']
        clusterID = batch['clusterID']   
        clusterID2 = batch['clusterID2']            
        cited = torch.cat([batch['title'], batch['abstract']], dim=1)
        nCited = torch.cat([batch['nTitle'], batch['nAbstract']], dim=1) #neighbor
        cited2 = torch.cat([batch['title2'], batch['abstract2']], dim=1)
        fCited = torch.cat([batch['fTitle'], batch['fAbstract']], dim=1)
        optimizer.zero_grad()
        pos_pred = model(context, clusterID, cited)
        neighbor_pred = model(context, clusterID, nCited)

        neg_pred1 = model(context, clusterID, fCited) # random
        neg_pred2 = model(context, clusterID, cited2)
        neg_pred3 = model(context, clusterID2, cited)
        loss1 = torch.mean(F.relu(ALPHA + pos_pred - neighbor_pred))
        loss2 = torch.mean(F.relu(ALPHA + neighbor_pred - neg_pred1))
        loss3 = torch.mean(F.relu(ALPHA + pos_pred - neg_pred2))
        loss4 = torch.mean(F.relu(ALPHA + pos_pred - neg_pred3))
        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num += ((pos_pred - neighbor_pred) < 0).sum()
        length += cited.shape[0]
    return epoch_loss / len(train_loader), float(num) / length

# %%
def secondEvaluate(model: nn.Module,
            train_loader: DataLoader,
            device = DEVICE):
    model.eval()
    epoch_loss = 0
    num = 0
    length = 0
    with torch.no_grad():
        for batch in train_loader:
            for key in batch.keys():
                if 'Raw' not in key:
                    batch[key] = batch[key].to(device)
            context = batch['context']
            clusterID = batch['clusterID']   
            clusterID2 = batch['clusterID2']            
            cited = torch.cat([batch['title'], batch['abstract']], dim=1)
            nCited = torch.cat([batch['nTitle'], batch['nAbstract']], dim=1)
            cited2 = torch.cat([batch['title2'], batch['abstract2']], dim=1)
            fCited = torch.cat([batch['fTitle'], batch['fAbstract']], dim=1)
            
            pos_pred = model(context, clusterID, cited)
            neighbor_pred = model(context, clusterID, nCited)

            neg_pred1 = model(context, clusterID, fCited) # random
            neg_pred2 = model(context, clusterID, cited2)
            neg_pred3 = model(context, clusterID2, cited)
            loss1 = torch.mean(F.relu(ALPHA + pos_pred - neighbor_pred))
            loss2 = torch.mean(F.relu(ALPHA + neighbor_pred - neg_pred1))
            loss3 = torch.mean(F.relu(ALPHA + pos_pred - neg_pred2))
            loss4 = torch.mean(F.relu(ALPHA + pos_pred - neg_pred3))
            loss = loss1 + loss2 + loss3 + loss4
            epoch_loss += loss.item()
            num += ((pos_pred - neighbor_pred) < 0).sum()
            length += cited.shape[0]
    return epoch_loss / len(train_loader), float(num) / length
# %%
def secondTraingProcess(model, train_loader, test_loader, optimizer, epoches=EPOCHES):
    if path.exists("runs/base"):
        shutil.rmtree("runs/base")
    if not path.exists("check_point"):
        makedirs("check_point/base")

    start = time.time()
    writer = SummaryWriter(log_dir='runs/base')
    for epoch in range(epoches):
        train_loss, train_acc = secondTrain(model, train_loader, optimizer)
        test_loss, test_acc = secondEvaluate(model, test_loader)
        writer.add_scalars('loss', {'train':train_loss, 'test':test_loss}, epoch)
        writer.add_scalars('accuracy', {'train':train_acc, 'test':test_acc}, epoch)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), "./check_point/base/base{}.pt".format(epoch))
        print('epoch {} finished, loss: {:.4f}, train accuracy: {:.4f}, test accuracy: {:.4f}, time: {}'.format(epoch, train_loss, train_acc, test_acc, timeSince(start))) 
# %%
smallMag = pd.read_pickle(SMALL_MAG)
# %%
samples = pd.read_pickle(SAMPLES)
# %%
rawSet = secondCitationSet(samples, smallMag)
trainSize = int(0.8 * len(rawSet))
testSize = len(rawSet) - trainSize

trainSet, testSet = torch.utils.data.random_split(rawSet, [trainSize, testSize], generator = torch.manual_seed(TORCH_SEED))

train_loader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# %%
model = net.Net().to(DEVICE)
model.load_state_dict(torch.load(f'./check_point/{CHECK_POINT}', map_location=DEVICE))
optimizer = torch.optim.Adam(model.parameters())
# %%
secondTraingProcess(model, train_loader, test_loader, optimizer)
# %%