##############################################################
#   Libraries
##############################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering


##############################################################
#   Variable Definition
##############################################################


##############################################################
#   Class Definition
##############################################################
class ReadData:
    def __init__(self, file_name):
        # Read Data
        self.data = np.loadtxt(fname=file_name, dtype=float)
        # Separate Class
        self.class0 = self.data[self.data[:, 0] == 0]
        self.class1 = self.data[self.data[:, 0] == 1]

    def plot(self, target):
        if target.lower() == "class0":
            plt.scatter(self.class0[:, 1], self.class0[:, 2], c=self.class0[:, 0], s=50, cmap="viridis")
            plt.title("Class 0 of the Training Data")
        elif target.lower() == "class1":
            plt.scatter(self.class1[:, 1], self.class1[:, 2], c=self.class1[:, 0], s=50, cmap="viridis")
            plt.title("Class 1 of the Training Data")
        else:
            plt.scatter(self.data[:, 1], self.data[:, 2], c=self.data[:, 0], s=50, cmap="viridis")
            plt.title("All Data")
        plt.show()

    def cluster(self):
        # Clustering
        clustering = SpectralClustering(n_clusters=8, assign_labels="discretize", random_state=0).fit(self.data)


##############################################################
#   Function Prototype
##############################################################
def athletic():
    data = ReadData(file_name="2d/train.txt")
    data.plot(target="class0")
    data.plot(target="class1")
    data.plot(target="All")
    print("Here")


##############################################################
#   Main Function
##############################################################
def main():
    print("Hello World!")
    athletic()


##############################################################
#    Main Function Runner
##############################################################
if __name__ == "__main__":
    main()
