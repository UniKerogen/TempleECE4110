##############################################################
#   Libraries
##############################################################
from homework4_supplement import *
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import fractional_matrix_power
import decimal
import sys

##############################################################
#   Variable Definition
##############################################################
DENOISELOOP = 10
TRAIN_POINTS = 10000  # Points per Class
EVALUATION_POINTS = 10000  # Points per Class
TRAIN_0_MEAN = np.array([0, 0])
TRAIN_0_COV = np.array([[2, 0], [0, 1]])
TRAIN_1_MEAN = np.array([1, 0])
TRAIN_1_COV = np.array([[1, 0], [0, 2]])


##############################################################
#   Class Definition
##############################################################
class MultiGaussian:
    # Input: mean and cov for each of the two class | points per class
    def __init__(self, mean0, cov0, mean1, cov1, point_num):
        self.mean0 = mean0
        self.mean1 = mean1
        self.cov0 = cov0
        self.cov1 = cov1
        self.point_num = point_num

        # Generate Data for Class 0 #
        x0t, y0t = np.random.multivariate_normal(mean=self.mean0, cov=self.cov0, size=self.point_num).T
        x0 = np.asarray(x0t).reshape(1, self.point_num).T
        y0 = np.asarray(y0t).reshape(1, self.point_num).T
        class0 = np.column_stack((x0, y0))
        mu = np.mean(class0, axis=0)
        for loop in range(0, DENOISELOOP):
            for row in range(0, self.point_num):
                class0[row, :] = class0[row, :] - mu
                x0[row, :] = x0[row, :] - mu[0]
                y0[row, :] = y0[row, :] - mu[1]
            mu = np.mean(class0, axis=0)
        self.class0 = np.asarray(class0)
        # Generate Data for Class 1 #
        x1, y1 = np.random.multivariate_normal(mean=self.mean1, cov=self.cov1, size=self.point_num).T
        x1 = np.asarray(x1).reshape(1, self.point_num).T
        y1 = np.asarray(y1).reshape(1, self.point_num).T
        class1 = np.column_stack((x1, y1))
        mu = np.mean(class1, axis=0) - self.mean1
        for loop in range(0, DENOISELOOP):
            for row in range(0, self.point_num):
                class1[row, :] = class1[row, :] - mu
                x1[row, :] = x1[row, :] - mu[0]
                y1[row, :] = y1[row, :] - mu[1]
            mu = np.mean(class1, axis=0) - self.mean1
        self.class1 = np.asarray(class1)
        # Combine all data
        x_total = np.row_stack((x0, x1))
        y_total = np.row_stack((y0, y1))
        self.class_total = np.row_stack((np.zeros((self.point_num, 1)), np.ones((self.point_num, 1))))
        # Generate overall data set
        self.data = np.column_stack((x_total, y_total))
        self.data_all = np.column_stack((np.column_stack((self.class_total, x_total)), y_total))

        # Rotated Data #
        # Class 0
        x0rt, y0rt = np.random.multivariate_normal(mean=self.mean0, cov=self.cov0, size=self.point_num).T
        x0r = np.asarray(x0rt).reshape(1, self.point_num).T
        y0r = np.asarray(y0rt).reshape(1, self.point_num).T
        d0r = np.sqrt(x0r ** 2 + y0r ** 2)
        x0r = d0r * np.sin(45)
        y0r = d0r * np.cos(45)
        self.class0_r = np.column_stack((x0r, y0r))
        # Class 1
        x1rt, y1rt = np.random.multivariate_normal(mean=self.mean1, cov=self.cov1, size=self.point_num).T
        x1r = np.asarray(x1rt).reshape(1, self.point_num).T
        y1r = np.asarray(y1rt).reshape(1, self.point_num).T
        d1r = np.sqrt(x1r ** 2 + y1r ** 2)
        x1r = d1r * np.sin(45)
        y1r = d1r * np.cos(45)
        self.class1_r = np.column_stack((x1r, y1r))
        # Combine Data
        x_r_total = np.row_stack((x0r, x1r))
        y_r_total = np.row_stack((y0r, y1r))
        self.data_r = np.column_stack((x_r_total, y_r_total))
        self.data_r_all = np.column_stack((np.column_stack((self.class_total, x_r_total)), y_r_total))

        # # PCA Data Generation - Different Covariance, aka Class Dependent #
        # self.pca_cd_sample_num = self.class0.shape[0]
        # pca_cd_feature_num = self.class0.shape[1]  # Dimension
        # # Classes #
        # pca_cd_c0_cov_temp = np.cov(self.class0.T, bias=True) * pca_cd_feature_num  # 2x2
        # pca_cd_c1_cov_temp = np.cov(self.class1.T, bias=True) * pca_cd_feature_num  # 2x2
        # pca_cd_c0_mean_temp = np.mean(self.class0, 0)  # 2x1
        # pca_cd_c1_mean_temp = np.mean(self.class1, 0)  # 2x1
        # pca_cd_c0_zm_data = self.class0
        # pca_cd_c1_zm_data = self.class1
        # pca_cd_c0_mu = pca_cd_c0_mean_temp
        # pca_cd_c1_mu = pca_cd_c1_mean_temp
        # for loop in range(0, DENOISELOOP):
        #     for row in range(0, self.pca_cd_sample_num):
        #         pca_cd_c0_zm_data[row, :] = pca_cd_c0_zm_data[row, :] - pca_cd_c0_mu
        #         pca_cd_c1_zm_data[row, :] = pca_cd_c1_zm_data[row, :] - pca_cd_c1_mu
        #     pca_cd_c0_mu = np.mean(pca_cd_c0_zm_data, 0)
        #     pca_cd_c1_mu = np.mean(pca_cd_c1_zm_data, 0)
        # self.pca_cd_c0_zm_cov = np.cov(pca_cd_c0_zm_data, axis=0)
        # self.pca_cd_c1_zm_cov = np.cov(pca_cd_c1_zm_data, axis=0)
        # eig_vals_c0, eig_vecs_c0 = np.linalg.eig(pca_cd_c0_zm_data)
        # eig_vals_c1, eig_vecs_c1 = np.linalg.eig(pca_cd_c1_zm_data)
        # self.pca_cd_class0 = np.dot(np.dot(fractional_matrix_power(np.diag(eig_vals_c0), -0.5), eig_vecs_c0.T), pca_cd_c0_zm_data.T).T
        # self.pca_cd_class1 = np.dot(np.dot(fractional_matrix_power(np.diag(eig_vals_c1), -0.5), eig_vecs_c1.T), pca_cd_c1_zm_data.T).T
        # self.pca_cd_class0_cov = np.cov(self.pca_cd_class0, bias=True) * pca_cd_feature_num
        # self.pca_cd_class1_cov = np.cov(self.pca_cd_class1, bias=True) * pca_cd_feature_num
        # self.pca_cd_data = np.row_stack((self.pca_cd_class0, self.pca_cd_class1))
        # self.pca_cd_data_all = np.column_stack((self.class_total, self.pca_cd_data))

        # PCA Data Generation - Same Covariance, aka Class Independent #
        self.pca_ci_sample_num = self.data.shape[0]
        pca_ci_feature_num = self.data.shape[1]  # Dimension
        pca_ci_cov_temp = np.cov(self.data.T, bias=True) * pca_ci_feature_num  # 2x2
        pca_ci_mean_temp = np.mean(self.data, 0)  # 2x1
        pca_ci_zm_data = self.data
        pca_ci_mu = pca_ci_mean_temp
        for loop in range(0, DENOISELOOP):
            for row in range(0, self.pca_ci_sample_num):
                pca_ci_zm_data[row, :] = pca_ci_zm_data[row, :] - pca_ci_mu
            pca_ci_mu = np.mean(pca_ci_zm_data, 0)
        self.pca_zm_cov = np.cov(pca_ci_zm_data.T, bias=True) * pca_ci_feature_num  # 2x2
        eig_vals, eig_vecs = np.linalg.eig(pca_ci_cov_temp)
        self.pca_ci_data = np.dot(np.dot(fractional_matrix_power(np.diag(eig_vals), -0.5), eig_vecs.T), pca_ci_zm_data.T).T
        self.pca_ci_class0 = self.pca_ci_data[0:int(self.pca_ci_sample_num/2), :]
        self.pca_ci_class1 = self.pca_ci_data[int(self.pca_ci_sample_num/2):, :]
        self.pca_ci_data_all = np.column_stack((self.class_total, self.pca_ci_data))
        self.pca_ci_cov = np.cov(self.pca_ci_data.T, bias=True) * pca_ci_feature_num
        self.pca_ci_mean = np.mean(self.pca_ci_data, 0)  # 2x1

    def plot(self, save=False, show=True):
        plt.scatter(self.class0[:, 0], self.class0[:, 1], color='b', alpha=0.5, label="class0")
        plt.scatter(self.class1[:, 0], self.class1[:, 1], color='r', alpha=0.5, label="class1")
        plt.legend(loc=2)
        plt.title("Multivarient Gaussian Plot")
        if save:
            plt.savefig('multi_gaussian_plot.png')
        if show:
            plt.show()

    def plot_r(self, save=False, show=True):
        plt.scatter(self.class0_r[:, 0], self.class0_r[:, 1], color='b', alpha=0.5, label="class0")
        plt.scatter(self.class1_r[:, 0], self.class1_r[:, 1], color='r', alpha=0.5, label="class1")
        plt.legend(loc=2)
        plt.title("Multivarient Gaussian Plot")
        if save:
            plt.savefig('multi_gaussian_plot.png')
        if show:
            plt.show()

    def plot_pca(self, save=False, show=True):
        plt.scatter(self.pca_ci_class0[:, 0], self.pca_ci_class0[:, 1], color='b', label="class0")
        plt.scatter(self.pca_ci_class1[:, 0], self.pca_ci_class1[:, 1], color='r', label="class1")
        plt.legend(loc=2)
        plt.title("Multivarient Gaussian PCA Plot")
        if save:
            plt.savefig("multi_gaussian_plot_pca.png")
        if show:
            plt.show()


class PCATheoreticalClassifier:
    # Input: PCA data with class identifier | points per class
    def __init__(self, train, class_point):
        self.data = train
        self.class_point = class_point
        self.feature_num = train.shape[1] - 1  # 2 features
        # Separate into 2 class
        self.class0 = train[0:int(class_point), 1:]
        self.class1 = train[int(class_point):, 1:]
        # Obtain the mean for each class
        self.class0_mean = np.mean(self.class0, 0).reshape(1, 2).T
        self.class1_mean = np.mean(self.class1, 0).reshape(1, 2).T
        # Obtain the original class
        self.class_list = train[:, 0]

    def classify(self, eva):
        distance0 = float(0)
        distance1 = float(0)
        for index in range(0, self.feature_num):
            distance0 = distance0 + float((self.class0_mean[index] - eva[index])**2)
            distance1 = distance1 + float((self.class1_mean[index] - eva[index])**2)

        return [distance0, distance1]

    def discover(self, eva):
        discovered = []
        dis_result = []
        for row in range(0, eva.shape[0]):
            [result0, result1] = self.classify(eva=eva[row, 1:])
            if result0 >= result1:
                discovered.append(1)
            else:
                discovered.append(0)
            dis_result.append([result0, result1])
        difference = [0, 0]
        for idx in range(0, np.asarray(discovered).shape[0]):
            # print("Comparing -> Classified", detected[idx], "to Class ", classes[idx], " ", detected[idx]==classes[idx])
            if discovered[idx] == self.class_list[idx]:
                difference[0] += 1
            else:
                difference[1] += 1
        error_rate = float(difference[1]) / float(np.asarray(discovered).shape[0]) * 100
        return error_rate

    # THIS IS CASE SPECIFIC FOR ONLY 2D MESHGRID
    def dec_boundary(self, x, y):
        distance0 = (self.class0_mean[0, 0] - x)**2 + (self.class0_mean[1, 0] - y)**2
        distance1 = (self.class1_mean[0, 0] - x)**2 + (self.class1_mean[1, 0] - y)**2

        return distance0-distance1

    def plot(self, eva, contour=False, show=True, save=False):
        f, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(eva.class0[:, 0], eva.class0[:, 1], color='b', alpha=0.5, label="class0")
        ax.scatter(eva.class1[:, 0], eva.class1[:, 1], color='r', alpha=0.5, label="class1")
        ax.legend(loc=2)
        plt.title("PCA Theoretical Classifier")
        # Generate Decision Surface
        if contour:
            x_vec = np.linspace(*ax.get_xlim())
            x, y = np.meshgrid(x_vec, x_vec)
            plt.contour(x, y, self.dec_boundary(x=x, y=y), levels=[0])
        # Save & Displace
        if show:
            plt.show()
        if save:
            plt.savefig("pca_theoretical_classifier.png")


class MaxLikelihoodClassifier:
    # Input: data with class identifier | points per class
    def __init__(self, train, class_point):
        self.data = train
        self.class_point = class_point
        self.feature_num = train.shape[1] - 1  # 2 feature
        # Obtain original class
        self.class_list = train[:, 0]
        # Separate into 2 class
        self.class0 = train[0:int(class_point), 1:]
        self.class1 = train[int(class_point):, 1:]
        # Obtain info of class0
        self.class0_cov = np.cov(self.class0.T, bias=True) * self.feature_num  # 2x2 Denormalized
        self.class0_mean = np.mean(self.class0, axis=0)  # TRAIN_0_MEAN  # 2x1
        self.class0_inv_cov = np.linalg.inv(self.class0_cov)
        self.class0_det_cov = np.linalg.det(self.class0_cov)
        self.class0_constant = - self.feature_num * 0.5 * np.log(2 * np.pi) - 0.5 * np.log(self.class0_det_cov)
        # Obtain info of class1
        self.class1_cov = np.cov(self.class1.T, bias=True) * self.feature_num  # 2x2 Denormalized
        self.class1_mean = np.mean(self.class1, axis=0)  # 2x1
        self.class1_inv_cov = np.linalg.inv(self.class1_cov)
        self.class1_det_cov = np.linalg.det(self.class1_cov)
        self.class1_constant = - self.feature_num * 0.5 * np.log(2 * np.pi) - 0.5 * np.log(self.class1_det_cov)

    def classify(self, eva, classifier=False, threshold=None, mean=None, log=True):
        # Determine Classifier
        if classifier:
            mu = self.class1_mean
            inv_cov = self.class1_inv_cov
            constant = self.class1_constant
            # print("Using classifier 1")
        else:
            mu = self.class0_mean
            inv_cov = self.class0_inv_cov
            constant = self.class0_constant
            # print("using classifier 0")
        # Classify
        if mean is None:
            mat_temp = np.asarray(eva) - np.asarray(mu)  # 26x1
            log_prob = - 0.5 * np.dot(np.dot(mat_temp.T, inv_cov), mat_temp)
            log_prob = constant + log_prob
            prob = np.exp(log_prob)
        else:
            mat_temp = np.asarray(eva) - np.asarray(mean)  # 26x1
            log_prob = - 0.5 * np.dot(np.dot(mat_temp.T, inv_cov), mat_temp)
            log_prob = constant + log_prob
            prob = np.exp(log_prob)

        if threshold is not None:
            return np.where(prob >= threshold, True, False)
        else:
            if log:
                return log_prob
            else:
                return prob

    def mean(self, classifier=False):
        if classifier:
            return self.class1_mean.reshape(1, 2).T
        else:
            return self.class0_mean.reshape(1, 2).T

    def dec_boundary(self, x, classifier=False, prior=None):
        if prior is not None:
            g = self.classify(eva=x, classifier=classifier, threshold=None, mean=self.mean(classifier=classifier), log=True) + np.log(prior)
        else:
            g = self.classify(eva=x, classifier=classifier, threshold=None, mean=self.mean(classifier=classifier), log=True)
        return g

    def detector(self, eva, prior0, prior1):
        prob_result = []
        detected = []
        for idx in range(0, eva.shape[0]):
            result0 = self.classify(eva=eva[idx, 1:], classifier=False, threshold=None, mean=None, log=True)
            result1 = self.classify(eva=eva[idx, 1:], classifier=True, threshold=None, mean=None, log=True)
            if result0 * prior0 >= result1 * prior1:
                detected.append(0)
            else:
                detected.append(1)
            prob_result.append([result0, result1])
        # Find out the correct rate
        comparison = [0, 0]
        for idx in range(0, np.asarray(detected).shape[0]):
            # print("Comparing -> Classified", detected[idx], "to Class ", classes[idx], " ", detected[idx]==classes[idx])
            if detected[idx] == self.class_list[idx]:
                comparison[0] += 1
            else:
                comparison[1] += 1
        error_rate = float(comparison[1]) / float(np.asarray(detected).shape[0]) * 100
        return error_rate

    def dec_bound(self, x, prior0, prior1):
        g0 = self.dec_boundary(x=x, classifier=False, prior=prior0)
        g1 = self.dec_boundary(x=x, classifier=True, prior=prior1)
        return g0 - g1

    def plot(self, class0, class1, prior0=None, prior1=None, contour=False, show=True, save=False):
        # Determine Prior
        if (prior1 is not None) and (prior1 is not None):
            prior0_cal = prior0
            prior1_cal = prior1
        else:
            prior0_cal = 0.5
            prior1_cal = 0.5
        # Plot
        plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(111)
        ax1.scatter(class0[:, 0], class0[:, 1], color='b', label="class0")
        ax1.scatter(class1[:, 0], class1[:, 1], color='r', label="class1")
        plt.title("Maximum Likelihood Classified Data Plot")
        plt.legend(loc=2)
        if contour:
            x_vec = np.linspace(*ax1.get_xlim())
            plt.contour(x_vec, x_vec, self.dec_bound(x=x_vec, prior0=prior0_cal, prior1=prior1_cal), levels=[0])
        if show:
            plt.show()
        if save:
            plt.savefig("max_likelihood_classifier.png")


##############################################################
#   Function Prototype
##############################################################


##############################################################
#   Main Function
##############################################################
def main():
    print("Hello World!")
    # Generate Train Data
    train_data = MultiGaussian(mean0=TRAIN_0_MEAN, cov0=TRAIN_0_COV, mean1=TRAIN_1_MEAN, cov1=TRAIN_1_COV,
                               point_num=TRAIN_POINTS)
    print("Train Data Generated!")
    eva_data = MultiGaussian(mean0=TRAIN_0_MEAN, cov0=TRAIN_0_COV, mean1=TRAIN_1_MEAN, cov1=TRAIN_1_COV,
                             point_num=EVALUATION_POINTS)
    print("Evaluation Data Generated!")

    # Q1 - Plot Original Data
    eva_data.plot(save=True, show=True)
    # Q2 - Plot PAC Data
    eva_data.plot_pca(save=True, show=True)
    # Q3 - PCA Theoretical Classifier via Distance
    pca_classified = PCATheoreticalClassifier(train=train_data.pca_ci_data_all, class_point=TRAIN_POINTS)
    error_rate = pca_classified.discover(eva=eva_data.pca_ci_data_all)
    print("The PCA Theoretical Error Rate is", error_rate, "%")
    pca_classified.plot(eva=eva_data, contour=True, show=True, save=False)
    # Q4 - maximum likelihood classifier Original
    max_classified = MaxLikelihoodClassifier(train=train_data.data_all, class_point=TRAIN_POINTS)
    error_original = max_classified.detector(eva=eva_data.data_all, prior0=0.5, prior1=0.5)
    print("The Maximum Likelihood Classified Error Rate is", error_original, "%")
    # Q5 - maximum likelihood classifier PCA
    pca_max_classified = MaxLikelihoodClassifier(train=train_data.pca_ci_data_all, class_point=TRAIN_POINTS)
    error_pca = pca_max_classified.detector(eva=eva_data.pca_ci_data_all, prior0=0.5, prior1=0.5)
    print("The Maximum Likelihood Classified PCA Error Rate is", error_pca, "%")
    pca_max_classified.plot(class0=eva_data.pca_ci_class0, class1=eva_data.pca_ci_class1, prior0=0.5, prior1=0.5,
                            contour=True, show=True, save=True)
    # # Q6 - Rotated Data
    # rotated_max_classified = MaxLikelihoodClassifier(train=train_data.data_r_all, class_point=TRAIN_POINTS)
    # error_rotated = rotated_max_classified.detector(eva=eva_data.data_r_all, prior0=0.5, prior1=0.5)
    # print("The Maximum Likelihood Classified PCA Error Rate is", error_rotated, "%")
    # rotated_max_classified.plot(class0=eva_data.class0_r, class1=eva_data.class1_r, prior0=0.5, prior1=0.5,
    #                             contour=True, show=True, save=True)


##############################################################
#    Main Function Runner
##############################################################
if __name__ == "__main__":
    main()
