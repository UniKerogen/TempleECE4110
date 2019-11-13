##############################################################
#   Libraries
##############################################################
from homework9_supplement import *
import numpy as np
import math
import threading as td
import time
import scipy
from scipy.spatial.distance import cdist
import multiprocessing as mp

##############################################################
#   Variable Definition
##############################################################
OVERLAP = -1
TRAIN_NUM = 10000
EVALUATION_NUM = 5000


##############################################################
#   Class Definition
##############################################################
# YinYang Data Generation Organize Class
class DataGeneration:
    def __init__(self, overlap=OVERLAP):
        self.scale = scale = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}
        # Train Class
        train_data = SetYinYang(scale=scale, N1=TRAIN_NUM, N2=TRAIN_NUM, overlap=overlap)
        self.train0 = train_data.Yin
        self.train1 = train_data.Yang
        self.train_class = np.row_stack((np.zeros((TRAIN_NUM, 1)), np.ones((TRAIN_NUM, 1))))
        self.train = np.row_stack((self.train0, self.train1))
        self.train_all = np.column_stack((self.train_class, self.train))
        # Evaluation Class
        eva_data = SetYinYang(scale=scale, N1=EVALUATION_NUM, N2=EVALUATION_NUM, overlap=overlap)
        self.eva0 = eva_data.Yin
        self.eva1 = eva_data.Yang
        self.eva_class = np.row_stack((np.zeros((EVALUATION_NUM, 1)), np.ones((EVALUATION_NUM, 1))))
        self.eva = np.row_stack((self.eva0, self.eva1))
        self.eva_all = np.column_stack((self.eva_class, self.eva))

    def graph(self):
        print("Generating Original Graphs")
        plt.figure(figsize=(16, 8))
        ax1 = plt.subplot(121)
        ax1.scatter(self.train0[:, 0], self.train0[:, 1], color='r', alpha=0.5)
        ax1.scatter(self.train1[:, 0], self.train1[:, 1], color='b', alpha=0.5)
        plt.title("Training Data Plot")
        ax2 = plt.subplot(122)
        ax2.scatter(self.eva0[:, 0], self.eva0[:, 1], color='r', alpha=0.5)
        ax2.scatter(self.eva1[:, 0], self.eva1[:, 1], color='b', alpha=0.5)
        plt.title("Evaluation Data Plot")


# Classification Class
class KNearestNeighbor:
    def __init__(self, data, k):
        # Training Data
        self.data = data
        self.train = data.train_all
        self.eva = data.eva_all
        self.k = k

    def get_neighbor(self, data_point_eva, k):
        # euclidean distance calculation
        dists = cdist(self.data.train, np.asarray(data_point_eva[1:]).reshape((1, 2)), 'sqeuclidean')
        # Assign euclidean distance to each corresponding points
        output = np.column_stack((self.train, dists))
        output = output[np.argsort(output[:, -1])]
        # Get neighbor
        neighbor = output[:k, :]
        return neighbor

    def predict(self, eva_data, k):
        # Get neighbor
        neighbor = self.get_neighbor(data_point_eva=eva_data, k=k)
        # Obtain the class output of the nearest neighbor
        if k == 1:
            output = neighbor[0]
        else:
            output = neighbor[:, 0]
        # Find the most frequent class
        (values, counts) = np.unique(output, return_counts=True)
        ind = np.argmax(counts)
        prediction = values[ind]
        return [eva_data[0], prediction]

    def classify(self):
        incident = False
        # Loop though all eva data
        for eva_data in self.eva:
            [original, detected] = self.predict(eva_data=eva_data, k=self.k)
            if not incident:
                result = [original, detected]
                incident = True
            else:
                result = np.row_stack((result, [original, detected]))
        # print("Classified")
        # Calculate Error Rate
        error = 0
        for row in result:
            if row[0] != row[1]:
                error += 1
        error_rate = error / result.shape[0] * 100
        return error_rate


class CoreThread(td.Thread):
    def __init__(self, overlap, k):
        td.Thread.__init__(self)
        self.overlap = overlap
        self.data = DataGeneration(overlap=overlap)
        self.KNN = KNearestNeighbor(data=self.data, k=k)
        self.k = k

    def run(self):
        start_time = time.time()
        error_rate = self.KNN.classify()
        end_time = time.time()
        print("The Error Rate with overlap of", self.overlap, "and k at", self.k, "is", error_rate, "% with runtime of",
              end_time - start_time, "seconds")


##############################################################
#   Function Prototype
##############################################################
def subcore(overlap_value, print_info=True):
    threads = []
    index = 0
    k = [2 ** x for x in range(0, 11)]
    for k_value in k:
        thread = CoreThread(overlap=overlap_value, k=k_value)
        if print_info:
            print("### Thread", index, "has been created ###")
        try:
            thread.start()
            threads.append(thread)
            if print_info:
                print("### Thread", index, "has been started ###")
        except:
            print("Unable to start thread", index, "with k value of", k_value)
        index += 1
    for t in threads:
        t.join()
        
def contest(print_info=False):
    overlap = np.linspace(-1, 1, 11)
    pool = mp.Pool(20)
    pool.map(subcore, overlap)
    pool.close()
    pool.join()
    
def rally(print_info=False):
    # Starting multiprocessors
    print("Starting Process ...")
    overlap = np.linspace(-1, 1, 11)
    # Start 10 cores
    for overlap_value in overlap:
        cores = []
        index = 0
        # Initialize Core
        core = mp.Process(target=subcore, args=(overlap_value, print_info))
        # Start Core
        core.start()
        if print_info:
            print("### Core has started ###")
        # Append Core
        cores.append(core)
        # Sleep
        time.sleep(0.1)
    for t in cores:
        t.join()
    

def race(print_info=False):
    # Overlap Value
    overlap = np.linspace(-1, 1, 11)
    # k in range of [1, 1024]
    k = [2 ** x for x in range(0, 11)]
    # Loop Thread
    for overlap_value in overlap:
        threads = []
        index = 0
        print("******* Now working with overlap at", overlap_value, "*******")
        for k_value in k:
            thread = CoreThread(overlap=overlap_value, k=k_value)
            if print_info:
                print("### Thread", index, "has been created ###")
            try:
                thread.start()
                threads.append(thread)
                if print_info:
                    print("### Thread", index, "has been started ###")
            except:
                print("Unable to start thread", index, "with k value of", k_value)
            index += 1
        for t in threads:
            t.join()

def athletic(overlap, k):
    # Generate Data
    data = DataGeneration(overlap=overlap)
    # Establish KNN
    KNN = KNearestNeighbor(data=data, k=k)
    error_rate = KNN.classify()
    # Return value
    return error_rate


def coach():
    incident = False
    # Overlap as a step of 0.2 from -1 to 1
    overlap = np.linspace(-1, 1, 11)
    # k in range of [1, 1024]
    k = [2**x for x in range(0, 11)]
    for value in overlap:
        for k_value in k:
            error = athletic(overlap=value, k=k_value)
            attach_value = [overlap, k, error]
            if not incident:
                result_list = attach_value
                incident = True
            else:
                result_list = np.row_stack((result_list, attach_value))
    print("Overlap | K | Error_rate (%)")
    print(result_list)


##############################################################
#   Main Function
##############################################################
def main():
    print("Hello World!")

    # Individual Test
    # start = time.time()
    # print("Error rate is", athletic(overlap=0, k=5), "%")
    # end = time.time()
    # print("Takes", end - start, "seconds")

    # Race
    # race(print_info=False)
    
    # Rally
    # rally(print_info=True)
    
    # Contest
    contest(print_info=False)


##############################################################
#    Main Function Runner
##############################################################
if __name__ == "__main__":
    main()
