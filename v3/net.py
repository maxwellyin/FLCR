# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import shutil
import time
import math
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from os import path, makedirs

# %%
from const import DEVICE, BERT_SIZE, CONTEXT_LENGTH, CITED_SIZE, ALPHA, EPOCHES, timeSince
# %%
class citationSet(Dataset):
    def __init__(self, samples: pd.DataFrame, smallMag:pd.DataFrame, transform=None):
        self.samples = samples
        self.smallMag = smallMag
        self.transform = transform

        self.smallMagTitleLength = len(smallMag)
        self.length = len(samples)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):      
        _sample = self.samples.iat[idx,0]
        n = random.randrange(len(_sample['titles']))
        title = _sample['titles'][n]
        abstract = _sample['abstracts'][n]
        paperID = _sample['paperIDs'][n]

        fakeTitle, fakeAbstract = self.smallMag.loc[random.randrange(self.smallMagTitleLength), ['titleEmb', 'abstractEmb']]
        sample = {'contextRaw': _sample['contextRaw'], 'context': _sample['context'], 'title': title, 'abstract': abstract, 'fTitle': fakeTitle, 'fAbstract': fakeAbstract, 'paperIDRaw': paperID}

        if self.transform:
            sample = self.transform(sample)

        return sample
# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_embedding = nn.Linear(CONTEXT_LENGTH*BERT_SIZE, 2*BERT_SIZE)
        self.cited_embedding = nn.Linear(CITED_SIZE*BERT_SIZE, 2*BERT_SIZE)

    def forward(self, context, cited):
        context_embedded = self.context_embedding(context)
        cited_embedded = self.cited_embedding(cited)
        x1 = (context_embedded - cited_embedded)**2
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
        context = batch['context']            
        cited = torch.cat([batch['title'], batch['abstract']], dim=1)
        fCited = torch.cat([batch['fTitle'], batch['fAbstract']], dim=1)
        optimizer.zero_grad()
        pos_pred = model(context, cited)
        neg_pred = model(context, fCited)
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
            context = batch['context']
            cited = torch.cat([batch['title'], batch['abstract']], dim=1)
            fCited = torch.cat([batch['fTitle'], batch['fAbstract']], dim=1)
            pos_pred = model(context, cited)
            neg_pred = model(context, fCited)
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