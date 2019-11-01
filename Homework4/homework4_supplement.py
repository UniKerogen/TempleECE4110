##############################################################
#   Libraries
##############################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import fractional_matrix_power
import decimal
import sys


##############################################################
#   Variable Definition
##############################################################


##############################################################
#   Class Definition
##############################################################
class MaxLikeClassifier:
    # Obtain train data to set up the environment
    def __init__(self, train):
        self.data_set = train
        self.sample_num = train.shape[0]
        self.feature_num = train.shape[1]  # Dimension
        self.cov = np.cov(train.T, bias=True) * self.feature_num  # 26x26 Denormalized
        self.inv_cov = np.linalg.inv(self.cov)
        self.det_cov = np.linalg.det(self.cov)
        self.mu = np.mean(train, 0)  # 26x1
        self.constant = - self.feature_num * 0.5 * np.log(2 * np.pi) - 0.5 * np.log(self.det_cov)

    # Take the data to classify based on trained data
    def classify(self, dev, threshold=None, mean=None, log=True):
        if mean is None:
            mat_temp = np.asarray(dev) - np.asarray(self.mu)  # 26x1
            log_prob = - 0.5 * np.dot(np.dot(mat_temp.T, self.inv_cov), mat_temp)
            log_prob = self.constant + log_prob
            prob = np.exp(log_prob)
        else:
            mat_temp = np.asarray(dev) - np.asarray(mean)  # 26x1
            log_prob = - 0.5 * np.dot(np.dot(mat_temp.T, self.inv_cov), mat_temp)
            log_prob = self.constant + log_prob
            prob = np.exp(log_prob)

        if threshold is not None:
            return np.where(prob >= threshold, True, False)
        else:
            if log:
                return log_prob
            else:
                return prob

    # Return mean
    def mean(self):
        return self.mu.reshape(1, 2).T

    # Decision boundary with equal prior
    def dec_boundary(self, x, prior=None):
        if prior is not None:
            g = self.classify(dev=x, threshold=None, mean=self.mean(), log=True) + np.log(prior)
        else:
            g = self.classify(dev=x, threshold= None, mean=self.mean(), log=True)
        return g


##############################################################
#   Function Prototype
##############################################################
# 2 class detector
def detector(classifier0, classifier1, data, prior0):
    # Detect Class
    prob_result = []
    detected = []
    for idx in range(0, data.shape[0]):
        result_0 = decimal.Decimal(classifier0.classify(dev=data[idx, 1:], threshold=None, mean=None, log=True))
        result_1 = decimal.Decimal(classifier1.classify(dev=data[idx, 1:], threshold=None, mean=None, log=True))
        if result_0 * decimal.Decimal(prior0) >= result_1 * decimal.Decimal(1-prior0):
            detected.append(0)
        else:
            detected.append(1)
        prob_result.append([result_0, result_1])
    # Find out the correct rate
    classes = data[:, 0]
    comparison = [0, 0]
    for idx in range(0, np.asarray(detected).shape[0]):
        # print("Comparing -> Classified", detected[idx], "to Class ", classes[idx], " ", detected[idx]==classes[idx])
        if detected[idx] == classes[idx]:
            comparison[0] += 1
        else:
            comparison[1] += 1
    error_rate = float(comparison[1]) / float(np.asarray(detected).shape[0]) * 100
    # print("Error Rate is", error_rate, "%")
    return error_rate


##############################################################
#   Main Function
##############################################################
def main():
    print("Hello World!")
    print("This is the supplement file for homework 4")


##############################################################
#    Main Function Runner
##############################################################
if __name__ == "__main__":
    main()
