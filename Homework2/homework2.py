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
PRIOR_STEP = 100


##############################################################
#   Class Prototype
##############################################################
# Equation Found -
# https://math.stackexchange.com/questions/892832/why-we-consider-log-likelihood-instead-of-likelihood-in-gaussian-distribution
# https://math.stackexchange.com/questions/869502/multivariate-gaussian-integral-over-positive-reals
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
        self.constant = decimal.Decimal(- self.feature_num * 0.5 * np.log(2 * np.pi) - 0.5 * np.log(self.det_cov))

    # Take the data to classify based on trained data
    def classify(self, dev, threshold=None):
        mat_temp = np.asarray(dev) - np.asarray(self.mu)  # 26x1
        log_prob = decimal.Decimal(- 0.5 * np.dot(np.dot(mat_temp.T, self.inv_cov), mat_temp))
        log_prob = decimal.Decimal(self.constant + log_prob)
        prob = decimal.Decimal(np.exp(log_prob))

        if threshold is not None:
            return np.where(prob >= threshold, True, False)
        else:
            return prob


class TheoreticalClassifier:
    def __init__(self, train):
        self.data_set = train
        self.sample_num = train.shape[0]
        self.feature_num = train.shape[1]  # Dimension
        self.cov = np.cov(train.T, bias=True) * self.feature_num  # 26x26 Denormalized
        self.mu = np.mean(train, 0)  # 26x1
        zm_data = self.data_set
        mu = self.mu
        for loop in range(0, 10):
            for row in range(0, self.sample_num):
                zm_data[row, :] = zm_data[row, :] - mu
            mu = np.mean(zm_data, 0)
        # print(np.mean(zm_data, 0))
        # print(zm_data[:,0])
        self.zm_cov = np.cov(zm_data, bias=True) * self.feature_num  # 26x26
        # Obtain the eigen-decomposition of the original covariance matrix
        eig_vals, eig_vecs = np.linalg.eig(self.cov)
        # print(np.linalg.inv(eig_vecs)-eig_vecs.T)
        self.new_data = np.dot(np.dot(fractional_matrix_power(np.diag(eig_vals), -0.5), eig_vecs.T),zm_data.T).T
        # print(self.new_data.shape)
        self.new_cov = np.cov(self.new_data.T, bias=True) * self.feature_num
        # print(self.new_cov)
        self.new_mean = np.mean(self.new_data, 0)  # 26x1
        # print(new_mean)

    # Calculate the theoretical error rate
    # https://plot.ly/ipython-notebooks/principal-component-analysis/
    def classify(self, dev):
        distance = float(0)
        for index in range(0, self.feature_num):
            distance = float(distance) + float((self.new_mean[index] - dev[index])**2)

        return distance


##############################################################
#   Function Prototype
#############################################################
# Open and read in the file
def open_file(file_name):
    # Read in File
    data = np.loadtxt(fname=file_name, dtype=float, unpack=True)
    data_width = data.shape[0]  # 26
    data_height = data.shape[1]  # 18936d
    # print("The size of ", file_name, "is", data.shape)

    # Separate Data Set
    classes = data[:][0]
    features = data[:][1:]
    class_0 = []
    class_1 = []
    [class_0, class_1] = class_separation(data)

    return [data_width, data_height, np.asarray(class_0), np.asarray(class_1), data]


# Separate Class 0 and 1 Features
def class_separation(data):
    class_0 = []
    class_1 = []
    for index in range(0, data.shape[1]):
        if data[0][index] == 0:
            class_0.append(data[1:, index])
        else:
            class_1.append(data[1:, index])
    return [class_0, class_1]


# 2 class detector
def detector(classifier0, classifier1, file_name, prior0):
    # Obtain data from file
    [width, height, class_0, class_1, data] = open_file(file_name)
    # Detect Class
    prob_result = []
    detected = []
    for idx in range(0, data.shape[1]):
        result_0 = decimal.Decimal(classifier0.classify(data[1:, idx]))
        result_1 = decimal.Decimal(classifier1.classify(data[1:, idx]))
        if result_0 * decimal.Decimal(prior0) >= result_1 * decimal.Decimal(1-prior0):
            detected.append(0)
        else:
            detected.append(1)
        prob_result.append([result_0, result_1])
    # print("Detect Result: ")
    # print(prob_result)
    # Find out the correct rate
    classes = data[:][0]
    comparison = [0, 0]
    for idx in range(0, np.asarray(detected).shape[0]):
        # print("Comparing -> Classified", detected[idx], "to Class ", classes[idx], " ", detected[idx]==classes[idx])
        if detected[idx] == classes[idx]:
            comparison[0] += 1
        else:
            comparison[1] += 1
    # print(comparison)
    # print("Detected Result: ")
    # print(detected)
    # print("Original Result: ")
    # print(classes)
    error_rate = float(comparison[1]) / float(np.asarray(detected).shape[0]) * 100
    # print("Error Rate is", error_rate, "%")
    return error_rate


def discover(classifier0, classifier1, file_name):
    # Obtain data from file
    [width, height, class_0, class_1, data] = open_file(file_name)
    # Detect Class
    discovered = []
    dis_result = []
    for idx in range(0, data.shape[1]):
        result_0 = decimal.Decimal(classifier0.classify(data[1:, idx]))
        result_1 = decimal.Decimal(classifier1.classify(data[1:, idx]))
        if result_0 >= result_1:
            discovered.append(0)
        else:
            discovered.append(1)
        dis_result.append([result_0, result_1])
    # print("Discover Result:")
    # print(dis_result)
    # Find out correct rate
    classes = data[:][0]
    difference = [0, 0]
    for idx in range(0, np.asarray(discovered).shape[0]):
        # print("Comparing -> Classified", detected[idx], "to Class ", classes[idx], " ", detected[idx]==classes[idx])
        if discovered[idx] == classes[idx]:
            difference[0] += 1
        else:
            difference[1] += 1
    # print(comparison)
    # print("Discovered Result: ")
    # print(discovered)
    # print("Original Result: ")
    # print(classes)
    error_rate = float(difference[1]) / float(np.asarray(discovered).shape[0]) * 100
    # print("Error Rate is", error_rate, "%")
    return error_rate


##############################################################
#   Main Function
##############################################################
def main():
    print("Hello World!")
    # Read in data from train.txt
    [train_width, train_height, train_class_0, train_class_1, train_data] = open_file("train.txt")
    # Train the classifier
    classifier_0 = MaxLikeClassifier(train=train_class_0)
    classifier_1 = MaxLikeClassifier(train=train_class_1)
    print("Maximum Likelihood Classifiers Set")

    # # Read in data from dev.txt
    file1 = "dev.txt"
    print("Detecting with ", file1, "at equal prior")
    error_50 = detector(classifier0=classifier_0, classifier1=classifier_1, file_name=file1, prior0=0.5)
    print("The error rate is", error_50, "%")
    file2 = "train.txt"
    print("Detecting with ", file2, "at equal prior")
    detector(classifier0=classifier_0, classifier1=classifier_1, file_name=file2, prior0=0.5)
    print("The error rate is", error_50, "%")

    # Plot Error Rate over Prior of a Class
    print("Generating Error Rate Plot over Prior of a Class")
    prior = np.linspace(0, 1, PRIOR_STEP)
    # print(prior)
    error_list = []
    for item in prior:
        error_temp = detector(classifier0=classifier_0, classifier1=classifier_1, file_name=file1, prior0=item)
        error_list.append(error_temp)
    # print(error_list)
    plt.plot(prior, np.asarray(error_list))
    title = "Class 0 Error Rate over Prior"
    plt.title(title)
    plt.ylabel("Error Rate")
    plt.xlabel("Class 0 Prior Rate")
    plt.show()

    # Calculate the theoretical error rate
    # Train Classifier
    classifier_0_t = TheoreticalClassifier(train=train_class_0)
    classifier_1_t = TheoreticalClassifier(train=train_class_1)
    print("Theoretical Classifier Set")

    # Compute the theoretical error rate
    print("Discovering with ", file1, "for theoretical error rate")
    error_t = discover(classifier0=classifier_0_t, classifier1=classifier_1_t, file_name=file1)
    print("The error rate is", error_t, "%")
    print("Discovering with ", file2, "for theoretical error rate")
    error_t = discover(classifier0=classifier_0_t, classifier1=classifier_1_t, file_name=file2)
    print("The error rate is", error_t, "%")


##############################################################
#    Main Function Runner
##############################################################
if __name__ == "__main__":
    main()
