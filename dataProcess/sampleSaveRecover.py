# sampleSaveEnhance 一次没有执行完(跑了8天)，只好写了个脚本恢复并且继续执行
# %%
import re
import pickle
# %%
BATCH_SIZE = 1000
# %%
with open('./dataout/slurm-57272075.out') as f:
    s = f.read()
# %%
idxes = re.findall(r'_upIdx:(\d+), time', s)
# %%
idxes2 = [int(int(i)/BATCH_SIZE) for i in idxes]
# %%
with open("../data/taskIdx.pkl", "rb") as f:
    tasks = pickle.load(f)
# %%
length = len(tasks) // BATCH_SIZE + 1
# %%
remainIdxes = list(set(range(length)) - set(idxes2))
# %% 最后一组不满一千1751514， 所以有idxes2里面有两个1751
for i in range(len(idxes2)):
    if idxes2[i] in idxes2[:i]:
        print(i)
        print(idxes2[i])
# %%
remainIdxes2 = sorted(remainIdxes)
# %%
# 以下copy自 sampleSaveEnhance.py，两处修改
# 删除os.remove(OUT_DB)这段代码
# p.map(dataProcess, range(length)) 改为 p.map(dataProcess, remainIdxes2)
# %%
from multiprocessing import Pool
import sqlite3
import pickle
import numpy as np
import datetime
import os
# %%
CS_PARSED_PAPER = "../../unarXive/data/csParsedPaper.db"
CITED_MAG = "../../unarXive/data/smallMag.db" 
CITING_MAG = "../../unarXive/data/citingMag.db"
OUT_DB = "../data/samplesEnhance.db"
REFS = "../../unarXive/data/refs.db"
BERT_SIZE = 768
LEFT_SCOPE = 3
REGHT_SCOPE = 3
CONTEXT_LENGTH = LEFT_SCOPE + REGHT_SCOPE + 1
BATCH_SIZE = 1000
NUM_WORKERS = 16
# %%
with open("../data/taskIdx.pkl", "rb") as f:
    tasks = pickle.load(f)
# %%
length = len(tasks) // BATCH_SIZE + 1
# %%
def dataProcess(i: int):
    _downIdx = (i)*BATCH_SIZE
    _upIdx = (i+1)*BATCH_SIZE
    _upIdx = min(_upIdx, len(tasks))
    samples = []
    for idx in range(_downIdx, _upIdx):       
        arxiveID = tasks[idx][0]
        sentencIdx = tasks[idx][1]
        downIdx = sentencIdx - LEFT_SCOPE
        upIdx = sentencIdx + REGHT_SCOPE + 1
            
        task = (arxiveID, downIdx, upIdx)
        connCsParsedPaper = sqlite3.connect(CS_PARSED_PAPER, timeout=10)
        curCsParsedPaper = connCsParsedPaper.cursor()
        curCsParsedPaper.execute(f"""SELECT sentence, sentenceEmb, citedPaperIdx FROM csParsedPaper
                                    WHERE arxiveID = ? and sentenceIndex >= ? and sentenceIndex < ?""", task)
        rows = curCsParsedPaper.fetchall()
        connCsParsedPaper.close()
        sentences = []
        sEmbs = []
        cIdxs = []
        for row in rows:
            sentences.append(row[0])
            sEmbs.append(pickle.loads(row[1]))
            cIdxs = cIdxs + pickle.loads(row[2])
        if len(sEmbs) < CONTEXT_LENGTH:
            sEmbs = sEmbs + [np.zeros([BERT_SIZE], dtype=np.float32)] * (CONTEXT_LENGTH - len(sEmbs))
        
        sentences2 = filter(lambda x: type(x) is str, sentences)
        sentences3 = ' '.join(sentences2)
        sEmb2 = np.concatenate(sEmbs)

        connRefs = sqlite3.connect(REFS, timeout=1000)
        curRefs = connRefs.cursor()
        curRefs.execute("""SELECT citing_mag_id FROM bibitem
                        WHERE citing_arxiv_id = ?""", (arxiveID,))
        rows2 = curRefs.fetchall()
        connRefs.close()
        citingID = rows2[0][0]
        citingID2 = int(citingID, 16)

        titleEmbs = []
        abstractEmbs = []
        paperIDs = []

        connCitingMag = sqlite3.connect(CITING_MAG, timeout=1000)
        curCitingMag = connCitingMag.cursor()

        curCitingMag.execute("""SELECT rowid, titleEmb FROM papers
                                WHERE paperID = ?""", (citingID2,))
        rows3 = curCitingMag.fetchall()
        if len(rows3) == 0:
            continue
        _, citingTitleEmb = rows3[0]        

        curCitingMag.execute("""SELECT rowid, abstractEmb FROM abstractIndex
                                        WHERE paperID = ?""", (citingID2,))
        rows4 = curCitingMag.fetchall()
        if len(rows4) == 0:
            continue
        _, citingAbstractEmb = rows3[0]          

        citingTitleEmb2 = pickle.loads(citingTitleEmb)
        citingAbstractEmb2 = pickle.loads(citingAbstractEmb)
        connCitingMag.close()

        connSmallMag = sqlite3.connect(CITED_MAG, timeout=1000)
        curSmallMag = connSmallMag.cursor()
        for cIdx in cIdxs:
            cIdx2 = int(cIdx, 16)

            curSmallMag.execute("""SELECT rowid, titleEmb FROM papers
                                        WHERE paperID = ?""", (cIdx2,))
            rows5 = curSmallMag.fetchall()
            if len(rows5) == 0:
                # print("bad id: {}".format(cIdx3))
                continue

            _, titleEmb = rows5[0]

            curSmallMag.execute("""SELECT rowid, abstractEmb FROM abstractIndex
                                        WHERE paperID = ?""", (cIdx2,))
            rows6 = curSmallMag.fetchall()
            if len(rows6) == 0:
                # print("bad id: {}".format(cIdx3))
                continue
            _, abstractEmb = rows6[0]

            titleEmb2 = pickle.loads(titleEmb)
            abstractEmb2 = pickle.loads(abstractEmb)
            titleEmbs.append(titleEmb2)
            abstractEmbs.append(abstractEmb2)
            paperIDs.append(cIdx2)
        connSmallMag.close()

        if len(titleEmbs) == 0:
            continue

        # the titles and abstracts one to one correspondence
        # titles, abstracts and paperIDs are all of cited paper.
        sample = {"citingID": citingID2, "contextRaw": sentences3, "context": sEmb2, "citingTitle":citingTitleEmb2, "citingAbstract":citingAbstractEmb2, "titles": titleEmbs, "abstracts": abstractEmbs, "paperIDs": paperIDs}   
        samples.append((pickle.dumps(sample),))

    connOut = sqlite3.connect(OUT_DB, timeout=1000)
    curOut = connOut.cursor()
    curOut.executemany("""INSERT INTO samples(sample)
              VALUES(?)""", samples)
    connOut.commit()
    connOut.close()
    print("_upIdx:{}, time:{}".format(_upIdx, datetime.datetime.now()))
# %%
print(datetime.datetime.now())
with Pool(NUM_WORKERS) as p:
    p.map(dataProcess, remainIdxes2)
print(datetime.datetime.now())