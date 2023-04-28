1. `train.py` train the model
2. `kdtree.py` build kdtree
3. `recall.py` compute recall mrr map nDGG.
4. `secondBuild.py` compute neighbors for cited papers in small mag via kdtree and generate new `smallMag.pkl`.
5. `secondtrain.py` employ negative and positive mining strategies to train the model. We need to change SMALL_MAG and CHECK_POINT in `const.py` 