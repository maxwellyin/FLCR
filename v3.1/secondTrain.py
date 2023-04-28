# 每次要检查是否修改 SMALL_MAG 和 CHECK_POINT
# %%
import pickle
import time
import sqlite3
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
from os import makedirs, path
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import net

# %%
from const import SAMPLES, SMALL_MAG, DEVICE, TORCH_SEED, BATCH_SIZE, NUM_WORKERS, ALPHA, EPOCHES, CHECK_POINT, timeSince
# %%
class secondCitationSet(Dataset):
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
        cur = self.conn.cursor()
        cur.execute("""SELECT *
                        FROM samples
                        WHERE rowid=?""", (idx+1,))
        rows = cur.fetchall()
        _sample = pickle.loads(rows[0][0])
        citingTitle = _sample['citingTitle']
        citingAbstract = _sample['citingAbstract']
        n = random.randrange(len(_sample['titles']))
        title = _sample['titles'][n]
        abstract = _sample['abstracts'][n]
        paperID = _sample['paperIDs'][n]

        neighbors = self.sortedSmallMag.loc[paperID, 'neighbors'][1:]
        neighborIdx = random.choice(neighbors)
        neighborTitle, neighborAbstract = self.smallMag.loc[neighborIdx, ['titleEmb', 'abstractEmb']]

        fakeTitle, fakeAbstract = self.smallMag.loc[random.randrange(self.smallMagTitleLength), ['titleEmb', 'abstractEmb']]
        sample = {'contextRaw': _sample['contextRaw'], 'context': _sample['context'], 'citingTitle': citingTitle, 'citingAbstract': citingAbstract, 'title': title, 'abstract': abstract,'nTitle':neighborTitle, 'nAbstract':neighborAbstract, 'fTitle': fakeTitle, 'fAbstract': fakeAbstract, 'paperIDRaw': paperID}

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
        context, citingTitle, citingAbstract = batch['context'], batch['citingTitle'], batch['citingAbstract']          
        metaCiting = torch.stack([citingTitle, citingAbstract], dim=1)    
        cited = torch.stack([batch['title'], batch['abstract']], dim=1)
        nCited = torch.stack([batch['nTitle'], batch['nAbstract']], dim=1)
        fCited = torch.stack([batch['fTitle'], batch['fAbstract']], dim=1)
        optimizer.zero_grad()
        pos_pred = model(context, metaCiting, cited)
        neighbor_pred = model(context, metaCiting, nCited)
        neg_pred = model(context, metaCiting, fCited)
        loss1 = torch.mean(F.relu(ALPHA + pos_pred - neighbor_pred))
        loss2 = torch.mean(F.relu(ALPHA + neighbor_pred - neg_pred))
        loss = loss1 + loss2
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
            context, citingTitle, citingAbstract = batch['context'], batch['citingTitle'], batch['citingAbstract']          
            metaCiting = torch.stack([citingTitle, citingAbstract], dim=1)    
            cited = torch.stack([batch['title'], batch['abstract']], dim=1)
            nCited = torch.stack([batch['nTitle'], batch['nAbstract']], dim=1)
            fCited = torch.stack([batch['fTitle'], batch['fAbstract']], dim=1)
            pos_pred = model(context, metaCiting, cited)
            neighbor_pred = model(context, metaCiting, nCited)
            neg_pred = model(context, metaCiting, fCited)
            loss1 = torch.mean(F.relu(ALPHA + pos_pred - neighbor_pred))
            loss2 = torch.mean(F.relu(ALPHA + neighbor_pred - neg_pred))
            loss = loss1 + loss2
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
rawSet = secondCitationSet(SAMPLES, smallMag)
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