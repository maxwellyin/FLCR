# change the inverted table format of abstract back to normal
# %%
import pickle
import sqlite3
import json
import datetime
# %%
SAMPLES = "../data/samplesPublic.db"
# %%
def reFormat(abstractIndex_:str):
    abstractIndex = json.loads(abstractIndex_)
    abstract = [31415926] * abstractIndex['IndexLength'] #just choose some abnormal value 31415926
    for key in abstractIndex['InvertedIndex'].keys():
        for idx in abstractIndex['InvertedIndex'][key]:
            abstract[idx] = key
    abstract2 = list(filter(lambda x: x != 31415926, abstract))
    return ' '.join(abstract2)
# %%
con = sqlite3.connect(SAMPLES)
cur= con.cursor()
# %%
cur.execute("select count(*) from samples")
rows = cur.fetchall()
length = rows[0][0]
# %%
for i in range(length):
    try:
        cur.execute("select sample from samples where rowid=?", (i+1,))
        rows = cur.fetchall()
        sam = pickle.loads(rows[0][0])
        sam['abstracts'] = [reFormat(abstract) for abstract in sam['abstracts']]
        sam2 = pickle.dumps(sam)
        cur.execute("update samples set sample = ? where rowid=?", (sam2, i+1))
    except json.JSONDecodeError as e:
        print(e)
    if i % 10000 == 0:
        con.commit()
        print(datetime.datetime.now())
con.commit()
# %%
