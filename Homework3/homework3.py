##############################################################
#   Libraries
##############################################################
from homework3_supplement import *


##############################################################
#   Variable Definition
##############################################################
OVERLAP = -1
TRAIN_NUM = 10000
EVALUATION_NUM = 5000


##############################################################
#   Class Definition
##############################################################


##############################################################
#   Function Prototype
##############################################################
# Generate YinYang Data
def gen_data(data_point, over_lap):
    scale = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}
    data = SetYinYang(scale=scale, N1=data_point, N2=data_point, overlap=over_lap)
    # print(data.Yang.shape[0])
    # print(data.Yang[1:])
    return [data.Yin, data.Yang]


# Generate class number
def gen_class(data_point):
    return np.row_stack((np.zeros((data_point, 1)), np.ones((data_point, 1))))


# Decision Boundary Calculation
def dec_bound(x, classifier0, classifier1, prior):
    g0 = classifier0.dec_boundary(x, prior=prior)
    g1 = classifier1.dec_boundary(x, prior=prior)
    return g0 - g1


# Purpose Runner
def athletic(over_lap, graph=True, save=True, indicate=True):
    # Train YinYang with 10000
    [yin_train, yang_train] = gen_data(data_point=TRAIN_NUM, over_lap=over_lap)
    classifier_0 = MaxLikeClassifier(train=yin_train)
    classifier_0_mean = classifier_0.mean()
    classifier_1 = MaxLikeClassifier(train=yang_train)
    classifier_1_mean = classifier_1.mean()
    if indicate:
        print("Classifier Set!")

    # Test on training set
    data_train = np.column_stack((gen_class(data_point=TRAIN_NUM), np.row_stack((yin_train, yang_train))))
    error_train = detector(classifier0=classifier_0, classifier1=classifier_1, data=data_train, prior0=0.5)
    if indicate:
        print("The error for training set is", error_train, "%")

    # Generate evaluation set of 5000
    [yin_eva, yang_eva] = gen_data(data_point=EVALUATION_NUM, over_lap=over_lap)
    data_eva = np.column_stack((gen_class(data_point=EVALUATION_NUM), np.row_stack((yin_eva, yang_eva))))
    error_eva = detector(classifier0=classifier_0, classifier1=classifier_1, data=data_eva, prior0=0.5)
    if indicate:
        print("The error for evaluation set is", error_eva, "%")

    # Generate Plot
    if graph:
        print("Generating Graphs")
        plt.figure(figsize=(16, 8))
        ax1 = plt.subplot(121)
        ax1.scatter(yin_train[:, 0], yin_train[:, 1], color='r', alpha=0.5)
        ax1.scatter(yang_train[:, 0], yang_train[:, 1], color='b', alpha=0.5)
        plt.title("Training Data Plot")
        x_vec = np.linspace(*ax1.get_xlim())
        plt.contour(x_vec, x_vec, dec_bound(x_vec, classifier_0, classifier_1, prior=0.5), levels=[0], cmap="Greys_r")

        ax2 = plt.subplot(122)
        ax2.scatter(yin_eva[:, 0], yin_eva[:, 1], color='r', alpha=0.5)
        ax2.scatter(yang_eva[:, 0], yang_eva[:, 1], color='b', alpha=0.5)
        x_vec = np.linspace(*ax2.get_xlim())
        plt.title("Evaluation Data Plot")
        plt.contour(x_vec, x_vec, dec_bound(x_vec, classifier_0, classifier_1, prior=0.5), levels=[0], cmap="Greys_r")

        # Show Plot
        if not save:
            plt.show()

        # Save Plot
        if save:
            save_name = "overlap_at_" + str(over_lap) + ".png"
            plt.savefig(save_name)
            print("Graph Saved!")

    return [error_train, error_eva]


##############################################################
#   Main Function
##############################################################
def main():
    print("Hello World!")
    athletic(OVERLAP, graph=True, save=True, indicate=True)

    # Obtain result for different overlap values
    # overlap = [-1, -0.25, -0.1, 0, 0.1, 0.25, 1]
    # result = np.zeros((len(overlap), 3))
    # for row in range(0, result.shape[0]):
    #     print("Currently computing with overlap value of", overlap[row])
    #     result[row][0] = overlap[row]
    #     [train, eva] = athletic(over_lap=overlap[row], graph=False, save=False, indicate=False)
    #     result[row][1] = train
    #     result[row][2] = eva
    # print("Overlap | train_error (%) | eva_error (%)")
    # print(result)

    print("Computation Complete!")


##############################################################
#    Main Function Runner
##############################################################
if __name__ == "__main__":
    main()
