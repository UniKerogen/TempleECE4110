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
OVERLAP = -1
TRAIN_NUM = 10000
EVALUATION_NUM = 5000


##############################################################
#   Class Definition
##############################################################
# Generate Yin Yang Graph
class SetYinYang:
    def __init__(self, scale, N1, N2, overlap):
        # declare variables
        self.overlap = overlap  # overlap parameter between [-1, 1]
        self.Yin = None  # Yin set of data
        self.Yang = None  # Yang set of data
        self.Yin_label = 0  # label for Yin class
        self.Yang_label = 1  # label for Yang class
        self.xpt = 0  # x position of a point
        self.ypt = 0  # y position of a point
        self.distance1 = 0.0  # Distance of point from origin
        self.distance2 = 0.0  # Distance from center of Yin
        self.distance3 = 0.0  # Distance from center of Yang
        self.radius1 = 0.0  # Acceptable radius for Yin
        self.radius2 = 0.0  # Acceptable radius for Yang

        # initialize variables
        # The boundary, mean and standard deviation of plot
        self.xmax = scale['xmax']
        self.xmin = scale['xmin']
        self.ymax = scale['ymax']
        self.ymin = scale['ymin']
        self.xmean = self.xmin + 0.5 * (self.xmax - self.xmin)
        self.ymean = self.ymin + 0.5 * (self.ymax - self.ymin)
        self.stddev_center = 1.5 * (self.xmax - self.xmin) / 2

        # Creating empty lists to save points' coordinates
        self.Yin = []
        self.Yang = []

        # Calculate the radius of each class on the plot
        self.radius1 = 1.5 * ((self.xmax - self.xmin) / 4)
        self.radius2 = 0.75 * ((self.xmax - self.xmin) / 4)

        # Number of samples in each class
        self.nYin = N1
        self.nYang = N2

        # Producing some random numbers based on Normal distribution and then
        # calculating the points distance to each class, choosing the closest
        # set.
        # The look will exit when both classes has been built up.
        nYin_counter = 0
        nYang_counter = 0
        while (nYin_counter < self.nYin) | (nYang_counter < self.nYang):
            # generate points with Normal distribution
            xpt = np.random.normal(self.xmean, self.stddev_center, 1)[0]
            ypt = np.random.normal(self.ymean, self.stddev_center, 1)[0]
            # calculate radius for each generated point
            distance1 = np.sqrt(xpt ** 2 + ypt ** 2)
            distance2 = np.sqrt(xpt ** 2 + (ypt + self.radius2) ** 2)
            distance3 = np.sqrt(xpt ** 2 + (ypt - self.radius2) ** 2)
            # decide which class each point belongs to
            if distance1 <= self.radius1:
                if (xpt >= -self.radius1) & (xpt <= 0):
                    if ((distance1 <= self.radius1) | (distance2 <= self.radius2)) & (distance3 > self.radius2):
                        if nYin_counter < self.nYin:
                            self.Yin.append([xpt, ypt])
                            nYin_counter += 1
                    elif nYang_counter < self.nYang:
                        self.Yang.append([xpt, ypt])
                        nYang_counter += 1
                if (xpt > 0.0) & (xpt <= self.radius1):
                    if ((distance1 <= self.radius1) | (distance3 <= self.radius2)) & (distance2 > self.radius2):
                        if nYang_counter < self.nYang:
                            self.Yang.append([xpt, ypt])
                            nYang_counter += 1
                    elif nYin_counter < self.nYin:
                        self.Yin.append([xpt, ypt])
                        nYin_counter += 1

        # Translate each sample in Yin and Yang from the origin to
        # the center of the plot.
        # For implementing overlap, the overlap parameter multiply to one of
        # the plot center points. So the overlap parameter interferes in
        # translation process.
        self.Yang = np.array(self.Yang) + np.array([self.xmean, self.ymean])
        self.Yin = np.array(self.Yin) + np.array([self.xmean, self.ymean]) * (1 + self.overlap)

    def print_out(self):
        # print the label and coordinate of each point to stdout
        for row in self.Yin:
            print(self.Yin_label, row[0], row[1])
        for row in self.Yang:
            print(self.Yang_label, row[0], row[1])

    def plot(self):
        # Save the plot to a PNG file in current directory.
        plt.scatter(self.Yang[:, 0], self.Yang[:, 1], color='r', alpha=0.5)
        plt.scatter(self.Yin[:, 0], self.Yin[:, 1], color='b', alpha=0.5)
        plt.savefig('YinYang.png')


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
        self.zm_cov = np.cov(zm_data, bias=True) * self.feature_num  # 26x26
        # Obtain the eigen-decomposition of the original covariance matrix
        eig_vals, eig_vecs = np.linalg.eig(self.cov)
        self.new_data = np.dot(np.dot(fractional_matrix_power(np.diag(eig_vals), -0.5), eig_vecs.T),zm_data.T).T
        self.new_cov = np.cov(self.new_data.T, bias=True) * self.feature_num
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
##############################################################


##############################################################
#   Main Function
##############################################################
def main():
    print("Hello World!")
    print("This is the supplemental file for homework 3")


##############################################################
#    Main Function Runner
##############################################################
if __name__ == "__main__":
    main()
