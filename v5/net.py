# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import shutil
import time
import sqlite3
import math
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from os import path, makedirs

# %%
from const import DEVICE, BERT_SIZE, CONTEXT_LENGTH, CITED_SIZE, EMB_SIZE, ALPHA, EPOCHES, timeSince
# %%
class citationSet(Dataset):
    def __init__(self, samples, smallMag:pd.DataFrame, transform=None):
        self.conn = sqlite3.connect(samples)
        self.smallMag = smallMag
        self.sortedSmallMag = smallMag[['paperID','neighbors']].set_index(['paperID']).sort_index()
        self.transform = transform

        self.smallMagTitleLength = len(smallMag)
        cur = self.conn.cursor()
        cur.execute(f"""SELECT count(*) FROM samples""")
        rows = cur.fetchall()
        self.length = rows[0][0] 
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):      
        fakeTitle, fakeAbstract = self.smallMag.loc[random.randrange(self.smallMagTitleLength), ['titleEmb', 'abstractEmb']]

        cur = self.conn.cursor()
        cur.execute("""SELECT *
                        FROM samples
                        WHERE rowid=?""", (idx+1,))
        rows = cur.fetchall()
        _sample = pickle.loads(rows[0][0])

        maxGroupID = _sample['groupIDs'][-1]

        if  maxGroupID> 0:
            id1, id2 = random.sample(range(maxGroupID+1), k=2)
            n1 = random.choice([i for i, x in enumerate(_sample['groupIDs']) if x == id1])
            n2 = random.choice([i for i, x in enumerate(_sample['groupIDs']) if x == id2])

            title = _sample['titles'][n1] # cited title
            abstract = _sample['abstracts'][n1] # cited abstract
            paperID = _sample['paperIDs'][n1] # cited paperID

            title2 = _sample['titles'][n2] # cited title
            abstract2 = _sample['abstracts'][n2] # cited abstract
        else:
            id1 = 0
            id2 = 12
            n = random.randrange(len(_sample['groupIDs']))
            title = _sample['titles'][n] # cited title
            abstract = _sample['abstracts'][n] # cited abstract
            paperID = _sample['paperIDs'][n] # cited paperID 
            title2 = fakeTitle
            abstract2 = fakeAbstract

        neighbors = self.sortedSmallMag.loc[paperID, 'neighbors'][1:]
        neighborIdx = random.choice(neighbors)
        neighborTitle, neighborAbstract = self.smallMag.loc[neighborIdx, ['titleEmb', 'abstractEmb']]
        
        sample = {'context': _sample['context'], 'clusterID': id1, 'title': title, 'abstract': abstract, 'nTitle':neighborTitle, 'nAbstract':neighborAbstract, 'clusterID2': id2, 'title2': title2, 'abstract2': abstract2, 'fTitle': fakeTitle, 'fAbstract': fakeAbstract, 'paperIDRaw': paperID}

        if self.transform:
            sample = self.transform(sample)

        return sample
# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.clusterEmbedding = nn.Embedding(700, EMB_SIZE)
        self.context_embedding = nn.Linear(CONTEXT_LENGTH*BERT_SIZE + EMB_SIZE, 2*BERT_SIZE)
        self.cited_embedding = nn.Linear(CITED_SIZE*BERT_SIZE, 2*BERT_SIZE)

    def forward(self, context, clusterID, cited):
        clusterEmb = self.clusterEmbedding(clusterID)
        context2 = torch.cat([context, clusterEmb], dim=1)
        context_embedded = self.context_embedding(context2)
        cited_embedded = self.cited_embedding(cited)
        x1 = (context_embedded - cited_embedded)**2
        x2 = (x1.sum(axis=1))**(1/2)
        return x2
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
        loss3 = torch.mean(F.relu(ALPHA + neighbor_pred - neg_pred2))
        loss4 = torch.mean(F.relu(ALPHA + neighbor_pred - neg_pred3))
        loss = 3*loss1 + loss2 + loss3 + loss4
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
            loss3 = torch.mean(F.relu(ALPHA + neighbor_pred - neg_pred2))
            loss4 = torch.mean(F.relu(ALPHA + neighbor_pred - neg_pred3))
            loss = 3*loss1 + loss2 + loss3 + loss4
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
# %% ---------------------------------------------------------------------
if __name__ == "__main__":
    from const import SMALL_MAG, DEVICE, SAMPLES
    smallMag = pd.read_pickle(SMALL_MAG)
    rawSet = citationSet(SAMPLES, smallMag)
    # Suprised. Can't do this cause this run random twice.
    print(rawSet[0]['clusterID'])
    print(rawSet[0]['clusterID2'])
    sample = rawSet[0]
    print(sample['clusterID'])
    print(sample['clusterID2'])