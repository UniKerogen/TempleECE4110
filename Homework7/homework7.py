##############################################################
#   Libraries
##############################################################
from homework7_supplement import *

##############################################################
#   Variable Definition
##############################################################
OVERLAP = 0.45
TRAIN_NUM = 10000
EVALUATION_NUM = 5000


##############################################################
#   Class Definition
##############################################################
class LinearDiscriminantAnalysis:
    def __init__(self, train_class0, train_class1, overlap="Unknown"):
        # Process Data for class0
        self.overlap = overlap
        self.train_class0 = train_class0.T
        self.class0_mean = np.mean(self.train_class0, axis=1).reshape(1, 2).T
        self.class0_cov = np.cov(self.train_class0, bias=False)  # * self.train_class0.shape[0]
        # Process Data for class1
        self.train_class1 = train_class1.T
        self.class1_mean = np.mean(self.train_class1, axis=1).reshape(1, 2).T
        self.class1_cov = np.cov(self.train_class1, bias=False)  # * self.train_class1.shape[0]

        # Within-class scatter matrix
        self.within_class_scatter = self.class0_cov + self.class1_cov
        # Between-class scatter matrix
        self.between_class_scatter = \
            (self.class0_mean - self.class1_mean).dot((self.class0_mean - self.class1_mean).T)

        # w matrix
        sw_inv = np.linalg.inv(self.within_class_scatter)
        self.w = sw_inv.dot((self.class0_mean - self.class1_mean).reshape(1, self.class0_mean.shape[0]).T)
        self.w_norm = self.w / np.linalg.norm(self.w) * (-1)

        # Compute LDA Projection
        eigenvalue, eigenvector = np.linalg.eig(np.dot(np.linalg.inv(self.within_class_scatter), self.between_class_scatter))
        w = np.linalg.inv(self.within_class_scatter).dot((self.class0_mean-self.class1_mean))
        # Sort and find the greatest eigenvalue
        pairs = [(np.abs(eigenvalue[index]), eigenvector[:, index]) for index in range(len(eigenvalue))]
        pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
        self.w_matrix = np.hstack((pairs[0][1].reshape(self.train_class0.shape[0], 1), pairs[1][1].reshape(self.train_class0.shape[0], 1))).real
        # LDA Data
        self.lda_class0 = np.array(self.train_class0.T.dot(self.w_matrix))
        self.lda_class1 = np.array(self.train_class1.T.dot(self.w_matrix))
        # Threshold c in Fisher's LDA
        # self.c2 = self.w_matrix.dot(1/2 * (self.class0_mean + self.class1_mean))
        self.c3 = self.w_norm.T.dot(0.5 * (self.class0_mean + self.class1_mean))

    # Classify
    def classify(self, eva_class0, eva_class1, debug=False):
        print("Classifying")
        # By observing the data, this specific data set is more discriminate on the x-axis
        # So the determination of c will be based on the x-axis location
        # Instead of using both x and y
        print("Warning: This classifier is very case specific")
        # Determine the orientation of the data
        left_class = 0
        right_class = 1
        # Debug Session
        if debug:
            print("class0 mean", self.class0_mean[0, 0])
            print("class1 mean", self.class1_mean[0, 0])
            print("Decision point", self.c3)
            print("left class", left_class)
            print("right class", right_class)
        # Establish error calculation
        error = 0
        # Class0 classification
        eva0_lda = np.array(eva_class0.dot(self.w_matrix))
        for x_value in eva0_lda[:, 0]:
            if x_value < self.c3:
                detected = left_class
            else:
                detected = right_class
            if detected != 0:
                error += 1
        # Class1 Classification
        eva1_lda = np.array(eva_class1.dot(self.w_matrix))
        for x_value in eva1_lda[:, 0]:
            if x_value < self.c3:
                detected = left_class
            else:
                detected = right_class
            if detected != 1:
                error += 1
        error_rate = error / (eva_class0.shape[0] + eva_class1.shape[0]) * 100
        return error_rate

    # Plot original data set
    def plot(self, data=None, save=False):
        # Set up variable
        if data is None:
            class0 = self.train_class0.T
            class1 = self.train_class1.T
            title1 = "Training Data Plot"
            title2 = "LDA for Training Data"
            lda0 = self.lda_class0
            lda1 = self.lda_class1
            save_name = "original_plot_" + str(self.overlap) + ".png"
        else:
            length = data.shape[0]
            class0 = data[0:int(length/2), :]
            class1 = data[int(length/2):length, :]
            lda0 = np.array(class0.dot(self.w_matrix))
            lda1 = np.array(class1.dot(self.w_matrix))
            title1 = "Evaluation Data Plot"
            title2 = "LDA For Evaluation Data"
            save_name = "eva_plot_" + str(self.overlap) + ".png"
        # Plot
        plt.figure(figsize=(16, 8))
        ax1 = plt.subplot(121)
        ax1.scatter(class0[:, 0], class0[:, 1], color='r', alpha=0.5)
        ax1.scatter(class1[:, 0], class1[:, 1], color='b', alpha=0.5)
        plt.title(title1)
        ax2 = plt.subplot(122)
        ax2.scatter(lda0[:, 0], lda0[:, 1], color='r', alpha=0.5)
        ax2.scatter(lda1[:, 0], lda1[:, 1], color='b', alpha=0.5)
        # Decision line via c for x-axis
        y = np.linspace(*ax2.get_ylim())
        ax2.scatter([self.c3] * y.shape[0], y, color="g")
        plt.title(title2)
        if save:
            plt.savefig(save_name)
            print("Plot saved!")
        else:
            plt.show()
        print("Plot has been generated")


##############################################################
#   Function Prototype
##############################################################
# Generate YinYang Data
def gen_data(data_point, over_lap):
    scale = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}
    data = SetYinYang(scale=scale, N1=data_point, N2=data_point, overlap=over_lap)
    return [data.Yin, data.Yang]


# Generate class number
def gen_class(data_point):
    return np.row_stack((np.zeros((data_point, 1)), np.ones((data_point, 1))))


def athletic(over_lap, indicate=True, plot=True, save=False, debug=False):
    [yin_train, yang_train] = gen_data(data_point=TRAIN_NUM, over_lap=over_lap)
    if indicate:
        print("Training data set")
    # yin_train = np.array([[4, 2], [2, 4], [2, 3], [3, 6], [4, 4]])
    # yang_train = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])
    # Set up LDA
    lda = LinearDiscriminantAnalysis(train_class0=yin_train, train_class1=yang_train, overlap=over_lap)
    # Generate Evaluation data
    [yin_eva, yang_eva] = gen_data(data_point=EVALUATION_NUM, over_lap=over_lap)
    if indicate:
        print("Evaluation data set")
    # Classification
    eva_error_rate = lda.classify(eva_class0=yin_eva, eva_class1=yang_eva, debug=debug)
    train_error_rate = lda.classify(eva_class0=yin_train, eva_class1=yang_train, debug=debug)
    if indicate:
        print("The training error rate is", train_error_rate, "%")
        print("The evaluation error rate is", eva_error_rate, "%")
    # Print Evaluation data
    if plot:
        lda.plot(data=None, save=save)
        eva_all = np.row_stack((yin_eva, yang_eva))
        lda.plot(data=eva_all, save=save)
    return [train_error_rate, eva_error_rate]


##############################################################
#   Main Function
##############################################################
def main():
    print("Hello World!")
    athletic(OVERLAP, indicate=True, plot=True, save=True, debug=True)
    # Obtain result for different overlap values
    overlap = [-1, -0.25, -0.1, 0, 0.1, 0.25, 1]
    result = np.zeros((len(overlap), 3))
    for row in range(0, result.shape[0]):
        print("Currently computing with overlap value of", overlap[row])
        result[row][0] = overlap[row]
        [train, eva] = athletic(over_lap=overlap[row], plot=False, save=False, indicate=False)
        result[row][1] = train
        result[row][2] = eva
    print("Overlap | train_error (%) | eva_error (%)")
    print(result)


##############################################################
#    Main Function Runner
##############################################################
if __name__ == "__main__":
    main()
