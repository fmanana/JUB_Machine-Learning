Project Summary
===============

Goals Achieved
--------------

The goals for this project was to run K-means clustering algorithm
on a small dataset (N=200) which represents handwritten digits and extract
conclusions on different K clustering and different initializations.

Lessons Learned
---------------

Our training points set contains different type of handwritten 1s and when
one tries to cluster those training points into more clusters, the more means
of different types are yielded. In other words, as K approaches the size
of the training set, we can see more clear categorization of 1s.
Additionally, different types of initialization in this case yield different
clusters. The reason for this is that our training set is not that big (only 200).
In the first three cases where K<<200 we can see different clusters for different
initialization. But once K=200, categorization of these 1s becomes more clear
and clusters do not differ depending on the type of initialization.
