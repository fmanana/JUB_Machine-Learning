Tutorial
========

Here is how to use the algorithm. Go to the source directory called "src"
and run "$ python3 K-means.py" in the terminal. Please check first whether
you fulfill all requirements for this project. You will be prompted to
enter K: for this version please enter only 1, 2, 3 or 200. After you
input K, the algorithm will run and you will be presented with a plot
from a random cluster. If you run the algorithm again, you might get
another cluster. If you want to check specific cluster, open the code
"K-means.py" and edit the last line where the plot is shown. For example if
you want to check the first cluster, then the line should be:

plt.imshow(cb_vectors[0,:].reshape(16,15))

If you want to check the second cluster, the line should be:

plt.imshow(cb_vectors[1,:].reshape(16,15))

and so on. 
