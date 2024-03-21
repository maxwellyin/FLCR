1. `train.py` train the model
2. `kdtree.py` build kdtree
3. `recall.py` compute recall mrr map nDGG.
4. `secondBuild.py` compute neighbors for cited papers in small mag via kdtree and generate new `smallMag.pkl`.
5. `secondtrain.py` employ negative and positive mining strategies to train the model. We need to change SMALL_MAG and CHECK_POINT in `const.py` 
6. `cluster.py` 根据老师的想法，选得分最高的一篇文章的最近的十个邻居为一类，做了对单一输入的推荐，类似demo，只是没有网页。
7. `clusterRecall.py` is based on `cluster.py` and compute the recall for test dataset.