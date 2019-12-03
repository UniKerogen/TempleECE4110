##############################################################
#   Libraries
##############################################################
from homework10_conda.homework10_supplement import *
from homework12_conda.lbg import *
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

##############################################################
#   Variable Definition
##############################################################


##############################################################
#   Class Definition
##############################################################
class KMeanCluster:
    def __init__(self, data, n_clusters=8, plot=False):
        self.train = data.train
        self.train_class0 = data.train0
        self.train_class1 = data.train1
        self.eva = data.eva
        self.eva_class = data.eva_class
        self.classifier = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10, random_state=0)
        # Fit data into KMeans for classifier
        self.classifier.fit(self.train)
        if plot:
            plt.scatter(self.train[:, 0], self.train[:, 1],
                        c=self.classifier.predict(X=self.train), s=50, cmap="viridis")
            plt.show()
        # Classify evaluation data
        self.result = self.classifier.cluster_centers_

    def distort(self, plot=False):
        # Compare to original class
        class0_mean = np.mean(self.train_class0, axis=0)
        class1_mean = np.mean(self.train_class1, axis=0)
        # Get number of correctness
        diff_class0 = np.sqrt(np.sum(np.square(self.result[0, :] - class0_mean)))
        diff_class1 = np.sqrt(np.sum(np.square(self.result[1, :] - class1_mean)))
        # Get error rate
        distortion = [diff_class0, diff_class1]
        return distortion

    def detector(self):
        # Obtain result from classifier
        class0 = self.result[0, :]
        class1 = self.result[1, :]
        # Euclidean Distance
        dists0 = cdist(self.eva, np.asarray(class0).reshape((1, 2)), 'sqeuclidean')
        dists1 = cdist(self.eva, np.asarray(class1).reshape((1, 2)), 'sqeuclidean')
        # Compare
        predicted = dists0 > dists1
        class_result = np.equal(predicted, self.eva_class)
        correct = np.sum(class_result)
        # Get error rate
        error = class_result.shape[0] - correct
        error_rate = error / class_result.shape[0] * 100

        return error_rate


class LindeBusoGrayCluster:
    def __init__(self, data, info=False):
        self.train = data.train
        self.train_class0 = data.train0
        self.train_class1 = data.train1
        self.eva = data.eva
        self.eva_class = data.eva_class
        cb, cb_abs_w, cb_rel_w = generate_codebook(self.train, 2)
        if info:
            for i, c in enumerate(cb):
                print('> %s, abs_weight=%d, rel_weight=%f' % (c, cb_abs_w[i], cb_rel_w[i]))
        result = np.asarray(cb)
        self.result = result[[1, 0]]

    def distort(self, info=False):
        # Compare to original class
        class0_mean = np.mean(self.train_class0, axis=0)
        class1_mean = np.mean(self.train_class1, axis=0)
        # Get number of correctness
        diff_class0 = np.sqrt(np.sum(np.square(self.result[0, :] - class0_mean)))
        diff_class1 = np.sqrt(np.sum(np.square(self.result[1, :] - class1_mean)))
        # Get error rate
        distortion = [diff_class0, diff_class1]
        return distortion

    def detector(self):
        # Obtain result from classifier
        class0 = self.result[0, :]
        class1 = self.result[1, :]
        # Euclidean Distance
        dists0 = cdist(self.eva, np.asarray(class0).reshape((1, 2)), 'sqeuclidean')
        dists1 = cdist(self.eva, np.asarray(class1).reshape((1, 2)), 'sqeuclidean')
        # Compare
        predicted = dists0 > dists1
        class_result = np.equal(predicted, self.eva_class)
        correct = np.sum(class_result)
        # Get error rate
        error = class_result.shape[0] - correct
        error_rate = error / class_result.shape[0] * 100

        return error_rate


##############################################################
#   Function Prototype
##############################################################
def athletic(overlap):
    km = KMeanCluster(data=DataGeneration(overlap=overlap), n_clusters=2, plot=True)
    distortion = km.distort(plot=True)
    print("The error with overlap of", overlap, "is", "{:.2f}".format(distortion[0]), "with class0 and",
          "{:.2f}".format(distortion[1]), "with class1")
    error_rate = km.detector()
    print("The error rate with overlap of", overlap, "is", "{:.2f}".format(error_rate), "%")

    lbg = LindeBusoGrayCluster(data=DataGeneration(overlap=overlap))
    distortion = lbg.distort(info=False)
    print("The error with overlap of", overlap, "is", "{:.2f}".format(distortion[0]), "with class0 and",
          "{:.2f}".format(distortion[1]), "with class1")
    error_rate = lbg.detector()
    print("The error rate with overlap of", overlap, "is", "{:.2f}".format(error_rate), "%")


##############################################################
#   Main Function
##############################################################
def main():
    print("Hello World!")
    overlap = np.linspace(-1, 1, 11)
    for item in overlap:
        athletic(overlap=item)
        print("*************************************")


##############################################################
#    Main Function Runner
##############################################################
if __name__ == "__main__":
    main()
