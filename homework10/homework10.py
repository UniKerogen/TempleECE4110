##############################################################
#   Libraries
##############################################################
from homework10_conda.homework10_supplement import *
import numpy as np
import threading as td
from sklearn.ensemble import RandomForestClassifier
import time


##############################################################
#   Variable Definition
##############################################################


##############################################################
#   Class Definition
##############################################################
class RandomForest:
    def __init__(self, data):
        self.train = data.train
        self.train_class = data.train_class
        self.eva = data.eva
        self.eva_class = data.eva_class
        self.classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

    def classify(self):
        # Fit data into random forest for classifier
        class_label = np.array(self.train_class[:]).reshape(self.train_class.shape[0])
        self.classifier.fit(X=self.train, y=class_label)
        # Classify evaluation data
        result = self.classifier.predict(X=self.eva)
        return result

    def detector(self):
        # Obtain result from classifier
        result = self.classify()
        # Compare to original class
        class_result = np.equal(result, self.eva_class.T)
        # Get number of correctness
        correct = np.sum(class_result)
        # Get error rate
        error = class_result.shape[1] - correct
        error_rate = error / class_result.shape[1] * 100
        return error_rate


class CoreThread(td.Thread):
    def __init__(self, overlap):
        td.Thread.__init__(self)
        self.overlap = overlap
        self.data = DataGeneration(overlap=overlap)
        self.random_forest = RandomForest(data=self.data)

    def run(self):
        start_time = time.time()
        error_rate = self.random_forest.detector()
        end_time = time.time()
        print("The Error Rate with overlap of", self.overlap, "is", "{:.2f}".format(error_rate), "% with runtime of",
              end_time - start_time, "seconds")


##############################################################
#   Function Prototype
##############################################################
def athletic(overlap):
    rf = RandomForest(data=DataGeneration(overlap=overlap))
    error_rate = rf.detector()
    print("The error rate with overlap of", overlap, "is", "{:.2f}".format(error_rate), "%")


def race(print_info=False):
    overlap = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    threads = []
    index = 0
    for overlap_value in overlap:
        thread = CoreThread(overlap=overlap_value)
        if print_info:
            print("### Thread", index, "has been created for overlap value of", overlap_value, "###")
        try:
            thread.start()
            threads.append(thread)
            if print_info:
                print("### Thread", index, "has been started for overlap value of", overlap_value, "###")
        except:
            print("Unable to start thread", index, "with overlap value of", overlap_value)
        index += 1
    for t in threads:
        t.join()


##############################################################
#   Main Function
##############################################################
def main():
    # Single performance
    athletic(overlap=0.0)

    # Race
    # race(print_info=False)


##############################################################
#    Main Function Runner
##############################################################
if __name__ == "__main__":
    main()
