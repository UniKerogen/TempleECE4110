
##############################################################
#   Libraries
##############################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mn


##############################################################
#   Variable Definition
##############################################################


##############################################################
#   Function Prototype
##############################################################
def open_file(file_name):
    # Read in File
    data = np.loadtxt(fname=file_name, dtype=float, unpack=True)
    data_width = data.shape[0]  # 26
    data_height = data.shape[1]  # 18936

    # Separate Data Set
    classes = data[:][0]
    features = data[:][1:]

    return [data_width, data_height, classes, features]


def get_data(file_name):
    # Get Data
    [data_width, data_height, classes, features] = open_file(file_name)

    # Calculate All Variance
    for loop0 in range(0, data_width-1):
        print("Current Loop ", loop0, " Current Var ", np.var(features[:][loop0]))
        print("Current Loop ", loop0, " Current Cov ", np.cov(features[:][loop0]))

    # Find the first max var event
    var_max_1 = np.var(features[:][0])
    var_max_1_column = 0
    for loop0 in range(1, data_width-1):
        var_max_1_temp = np.var(features[:][loop0])
        if var_max_1_temp > var_max_1:
            var_max_1 = var_max_1_temp
            var_max_1_column = loop0
    # print("max_var_1 = ", var_max_1)
    # print("max_var-1_column = ", var_max_1_column)

    # Find the second max var event
    var_max_2 = 0
    var_max_2_column = 0
    for loop1 in range(0, data_width-1):
        var_max_2_temp = np.var(features[:][loop1])
        if loop1 != var_max_1_column:
            if var_max_1_column == 0:
                var_max_2_column = 1
                var_max_2 = np.var(features[:][1])
            else:
                if var_max_2_temp > var_max_2:
                    var_max_2 = var_max_2_temp
                    var_max_2_column = loop1
    # print("max_var_2 = ", var_max_2)
    # print("max_var_2_column = ", var_max_2_column)

    # Repack Data
    x_axis = features[:][24]
    y_axis = features[:][25]

    # Return Value
    return [x_axis, y_axis, classes]


def plot_scatter(classes, x_axis, y_axis, plot_location):
    # Configure Color
    color_t = [[1, 0, 0], [0, 0, 1]]
    color = [color_t[int(loop3)] for loop3 in classes]

    # Plot
    plt.subplot(plot_location)
    plt.scatter(x_axis, y_axis, c=color)
    plt.title("Class Overlay plot")


def plot_data(file_name):
    # Obtain Axis and Class Data
    [x_axis, y_axis, classes] = get_data(file_name)

    # Configure Figure
    plt.figure()
    fig_scatter = 111

    # Plot the Data
    plot_scatter(classes, x_axis, y_axis, fig_scatter)

    # Show Additional Calculation
    fig_class_0 = 111
    fig_class_1 = 111
    gaussian(classes, x_axis, y_axis, file_name, fig_class_0, fig_class_1)

    # Save Figure
    # plt.show()
    # plt.savefig("Q2 - Greatest Overlap.png")


def gaussian_plot(unique1, unique2, name, file_name, fig_class):
    # Variable Configuration
    [mean1, std1, var1] = unique1
    [mean2, std2, var2] = unique2

    # Plot Configuration
    plt.subplot(fig_class)

    # Gaussian Plot Preparation
    number_of_points = 500000
    point_selection = 100

    # Bar Plot
    s1 = np.random.normal(mean1, std1, number_of_points)
    count1, bins1, ignored1 = plt.hist(s1, point_selection, density=True, color='white')
    s2 = np.random.normal(mean2, std2, number_of_points)
    count2, bins2, ignored2 = plt.hist(s2, point_selection, density=True, color='white')
    # Curve Plot
    gaussian_expression1 = 1 / np.sqrt(2 * np.pi * var1) * np.exp(-((bins1 - mean1) ** 2) / (2 * var1))
    gaussian_expression2 = 1 / np.sqrt(2 * np.pi * var2) * np.exp(-((bins2 - mean2) ** 2) / (2 * var2))
    plt.plot(bins1, gaussian_expression1, linewidth=1, color='red', label='class0')
    plt.plot(bins2, gaussian_expression2, linewidth=1, color='blue', label='class1')

    # Plot Information
    title = "".join((file_name, " - ", name))
    plt.title(title)
    plt.legend(loc='upper right')


def gaussian(classes, set_0, set_1, file_name, fig_feature_1, fig_feature_2):
    # Separate Class
    class_0 = []
    class_0_f1 = []
    class_0_f2 = []
    class_1 = []
    class_1_f1 = []
    class_1_f2 = []
    for loop0 in range(0, len(classes)):
        if classes[loop0] == 0:
            class_0 = np.append(np.append(class_0, set_0[loop0]), set_1[loop0])
            class_0_f1 = np.append(class_0_f1, set_0[loop0])
            class_0_f2 = np.append(class_0_f2, set_1[loop0])
        else:
            class_1 = np.append(np.append(class_1, set_0[loop0]), set_1[loop0])
            class_1_f1 = np.append(class_1_f1, set_0[loop0])
            class_1_f2 = np.append(class_1_f2, set_1[loop0])

    # Create Gaussian Distribution Module
    mean_multi_0 = np.array([np.mean(class_0_f1), np.mean(class_0_f2)])
    cov_multi_0 = np.cov(class_0_f1, class_0_f2)
    mean_multi_1 = [np.mean(class_1_f1), np.mean(class_1_f2)]
    cov_multi_1 = np.cov(class_1_f1, class_1_f2)
    gaussian_multi(mean_multi_0, cov_multi_0, [1, 0, 0], fig_feature_1, class_name="Class 0")
    gaussian_multi(mean_multi_1, cov_multi_1, [0, 0, 1], fig_feature_2, class_name="Class 1")
    plt.savefig("Q2 - Greatest Overlap.png")


def gaussian_multi(mean, cov, color, plot_location, class_name):
    # Generate Random Number
    x, y = np.mgrid[-0.1:0.1:0.001, -0.1:0.1:0.001]
    pos = np.dstack((x, y))
    rv = mn(mean, cov)
    # Plot
    plot = plt.subplot(plot_location)
    plot.contour(x, y, rv.pdf(pos))
    # plot.title("TBD")


##############################################################
#   Main Function
##############################################################
def main():
    print("Hello World!")
    plot_data("train.txt")


##############################################################
#    Main Function Runner
##############################################################
if __name__ == "__main__":
    main()
