# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pandas as pd
import shutil
import time
import sqlite3
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from os import path, makedirs

# %%
from const import DEVICE, BERT_SIZE, CONTEXT_LENGTH, CITING_SIZE, CITED_SIZE, ALPHA, EPOCHES, timeSince
# %%
class citationSet(Dataset):
    def __init__(self, samples, smallMag:pd.DataFrame, transform=None):
        self.conn = sqlite3.connect(samples)
        self.smallMag = smallMag
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

        fakeTitle, fakeAbstract = self.smallMag.loc[random.randrange(self.smallMagTitleLength), ['titleEmb', 'abstractEmb']]
        sample = {'contextRaw': _sample['contextRaw'], 'context': _sample['context'], 'citingTitle': citingTitle, 'citingAbstract': citingAbstract, 'title': title, 'abstract': abstract, 'fTitle': fakeTitle, 'fAbstract': fakeAbstract, 'paperIDRaw': paperID}

        if self.transform:
            sample = self.transform(sample)

        return sample

# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.citing_embedding = nn.Linear((CONTEXT_LENGTH+CITING_SIZE)*BERT_SIZE, 2*BERT_SIZE)
        self.cited_embedding = nn.Linear(CITED_SIZE*BERT_SIZE, 2*BERT_SIZE)
        self.p = nn.parameter.Parameter(torch.ones(CITING_SIZE))
        self.q = nn.parameter.Parameter(torch.ones(CITED_SIZE))

    def forward(self, context, meta_citing, meta_cited):
        meta_citing_= meta_citing.clone()
        meta_cited_ = meta_cited.clone()
        for i in range(CITING_SIZE):
            meta_citing_[:,i,:] = meta_citing[:,i,:] * self.p[i]               
        for i in range(CITED_SIZE):
            meta_cited_[:,i,:] = meta_cited[:,i,:] * self.q[i]
        meta_citing = meta_citing_.reshape(meta_citing_.shape[0], -1)
        meta_cited = meta_cited_.reshape(meta_cited_.shape[0], -1) 
        citing = torch.cat([context,meta_citing], dim = 1)
        citing_embedded = self.citing_embedding(citing)
        cited_embedded = self.cited_embedding(meta_cited)
        x1 = (citing_embedded - cited_embedded)**2
        x2 = (x1.sum(axis=1))**(1/2)
        return x2
# %%
def train(model: nn.Module, train_loader, optimizer, device = DEVICE):
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
        fCited = torch.stack([batch['fTitle'], batch['fAbstract']], dim=1)
        optimizer.zero_grad()
        pos_pred = model(context, metaCiting, cited)
        neg_pred = model(context, metaCiting, fCited)
        loss = torch.mean(F.relu(ALPHA + pos_pred - neg_pred))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num += ((pos_pred - neg_pred) < 0).sum()
        length += cited.shape[0]
    return epoch_loss / len(train_loader), float(num) / length
# %%
def evaluate(model: nn.Module,
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
            fCited = torch.stack([batch['fTitle'], batch['fAbstract']], dim=1)
            pos_pred = model(context, metaCiting, cited)
            neg_pred = model(context, metaCiting, fCited)
            loss = torch.mean(F.relu(ALPHA + pos_pred - neg_pred))
            epoch_loss += loss.item()
            num += ((pos_pred - neg_pred) < 0).sum()
            length += cited.shape[0]
    return epoch_loss / len(train_loader), float(num) / length
# %%
def traing_process(model, train_loader, test_loader, optimizer, epoches=EPOCHES):
    if path.exists("runs/base"):
        shutil.rmtree("runs/base")
    if not path.exists("check_point"):
        makedirs("check_point/base")

    start = time.time()
    writer = SummaryWriter(log_dir='runs/base')
    for epoch in range(epoches):
        train_loss, train_acc = train(model, train_loader, optimizer)
        test_loss, test_acc = evaluate(model, test_loader)
        writer.add_scalars('loss', {'train':train_loss, 'test':test_loss}, epoch)
        writer.add_scalars('accuracy', {'train':train_acc, 'test':test_acc}, epoch)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), "./check_point/base/base{}.pt".format(epoch))
        print('epoch {} finished, loss: {:.4f}, train accuracy: {:.4f}, test accuracy: {:.4f}, time: {}'.format(epoch, train_loss, train_acc, test_acc, timeSince(start))) 

