# This version doesn't contain citing title and abstract.
# %%
from multiprocessing import Pool
import sqlite3
import pickle
import numpy as np
import datetime
import os
# %%
CS_PARSED_PAPER = "../../unarXive/csParsedPaper.db"
SMALL_MAG = "../../unarXive/smallMag.db"
OUT_DB = "../data/smallSamples.db"
BERT_SIZE = 768
LEFT_SCOPE = 3
REGHT_SCOPE = 3
CONTEXT_LENGTH = LEFT_SCOPE + REGHT_SCOPE + 1
BATCH_SIZE = 1000
NUM_WORKERS = 16
# %%
if os.path.exists(OUT_DB):
    os.remove(OUT_DB)
os.system(f'sqlite3 {OUT_DB}<create.sql')
# %%
with open("./taskIdx.pkl", "rb") as f:
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
        titleEmbs = []
        abstractEmbs = []
        paperIDs = []

        connSmallMag = sqlite3.connect(SMALL_MAG, timeout=10)
        curSmallMag = connSmallMag.cursor()
        for cIdx in cIdxs:
            cIdx2 = int(cIdx, 16)

            curSmallMag.execute("""SELECT rowid, titleEmb FROM papers
                                        WHERE paperID = ?""", (cIdx2,))
            rows2 = curSmallMag.fetchall()
            if len(rows2) == 0:
                # print("bad id: {}".format(cIdx3))
                continue

            _, titleEmb = rows2[0]

            curSmallMag.execute("""SELECT rowid, abstractEmb FROM abstractIndex
                                        WHERE paperID = ?""", (cIdx2,))
            rows3 = curSmallMag.fetchall()
            if len(rows3) == 0:
                # print("bad id: {}".format(cIdx3))
                continue
            _, abstractEmb = rows3[0]

            titleEmb2 = pickle.loads(titleEmb)
            abstractEmb2 = pickle.loads(abstractEmb)
            titleEmbs.append(titleEmb2)
            abstractEmbs.append(abstractEmb2)
            paperIDs.append(cIdx2)
        connSmallMag.close()

        if len(titleEmbs) == 0:
            continue

        # the titles and abstracts one to one correspondence
        # paperIDs is IDs of cited paper.
        sample = {"contextRaw": sentences3, "context": sEmb2, "titles": titleEmbs, "abstracts": abstractEmbs, "paperIDs": paperIDs}   
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
    p.map(dataProcess, range(length))
    # p.map(dataProcess, range(10)) # size = input*1000
print(datetime.datetime.now())
# %%
