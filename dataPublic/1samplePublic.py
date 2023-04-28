# 基于sampleEnhance.py
# 最终发布版
# %%
from multiprocessing import Pool
import sqlite3
import pickle
import re
import datetime
import os
# %%
def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError as e:
        print(e)
# %%
CS_PARSED_PAPER = "../../unarXive/data/out/csParsedPaper.db"
CITED_MAG = "../../unarXive/data/out/smallMag.db" 
CITING_MAG = "../../unarXive/data/out/citingMag.db"
OUT_DB = "../data/samplesPublic.db"
REFS = "../../unarXive/data/refs.db"
BERT_SIZE = 768
LEFT_SCOPE = 3
REGHT_SCOPE = 3
CONTEXT_LENGTH = LEFT_SCOPE + REGHT_SCOPE + 1
BATCH_SIZE = 1000
NUM_WORKERS = available_cpu_count()
# %%
if os.path.exists(OUT_DB):
    os.remove(OUT_DB)
os.system(f'sqlite3 {OUT_DB}<create.sql')
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
        curCsParsedPaper.execute(f"""SELECT sentence, citedPaperIdx FROM csParsedPaper
                                    WHERE arxiveID = ? and sentenceIndex >= ? and sentenceIndex < ?""", task)
        rows = curCsParsedPaper.fetchall()
        connCsParsedPaper.close()
        sentences = []
        cIdxs = []
        for row in rows:
            sentences.append(row[0])
            cIdxs = cIdxs + pickle.loads(row[1])
        
        sentences2 = filter(lambda x: type(x) is str, sentences)
        sentences3 = ' '.join(sentences2)

        connRefs = sqlite3.connect(REFS, timeout=1000)
        curRefs = connRefs.cursor()
        curRefs.execute("""SELECT citing_mag_id FROM bibitem
                        WHERE citing_arxiv_id = ?""", (arxiveID,))
        rows2 = curRefs.fetchall()
        connRefs.close()
        citingID = rows2[0][0]
        citingID2 = int(citingID, 16)

        titles = []
        abstracts = []
        paperIDs = []

        connCitingMag = sqlite3.connect(CITING_MAG, timeout=1000)
        curCitingMag = connCitingMag.cursor()

        curCitingMag.execute("""SELECT rowid, paperTitle FROM papers
                                WHERE paperID = ?""", (citingID2,))
        rows3 = curCitingMag.fetchall()
        if len(rows3) == 0:
            continue
        _, citingTitle = rows3[0]        

        curCitingMag.execute("""SELECT rowid, abstractIndex FROM abstractIndex
                                        WHERE paperID = ?""", (citingID2,))
        rows4 = curCitingMag.fetchall()
        if len(rows4) == 0:
            continue
        _, citingAbstract = rows3[0]          

        connCitingMag.close()

        connSmallMag = sqlite3.connect(CITED_MAG, timeout=1000)
        curSmallMag = connSmallMag.cursor()
        for cIdx in cIdxs:
            cIdx2 = int(cIdx, 16)

            curSmallMag.execute("""SELECT rowid, paperTitle FROM papers
                                        WHERE paperID = ?""", (cIdx2,))
            rows5 = curSmallMag.fetchall()
            if len(rows5) == 0:
                # print("bad id: {}".format(cIdx3))
                continue

            _, title = rows5[0]

            curSmallMag.execute("""SELECT rowid, abstractIndex FROM abstractIndex
                                        WHERE paperID = ?""", (cIdx2,))
            rows6 = curSmallMag.fetchall()
            if len(rows6) == 0:
                # print("bad id: {}".format(cIdx3))
                continue
            _, abstract = rows6[0]

            titles.append(title)
            abstracts.append(abstract)
            paperIDs.append(cIdx2)
        connSmallMag.close()

        if len(titles) == 0:
            continue

        # the titles and abstracts one to one correspondence
        # titles, abstracts and paperIDs are all of cited paper.
        sample = {"citingID": citingID2, "context": sentences3, "citingTitle":citingTitle, "citingAbstract":citingAbstract, "titles": titles, "abstracts": abstracts, "paperIDs": paperIDs}   
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
