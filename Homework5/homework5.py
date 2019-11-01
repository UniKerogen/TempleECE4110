##############################################################
#   Libraries
##############################################################
import numpy as np
import matplotlib.pyplot as plt

##############################################################
#   Variable Definition
##############################################################
ACTUAL_MEAN = 1
ACTUAL_VAR = 1
POINTS = 10 ** 6
SET_NUM = 10


##############################################################
#   Class Definition
##############################################################
# Class Data Generation
class DataGen:
    def __init__(self, mean=ACTUAL_MEAN, var=ACTUAL_VAR, points=POINTS):
        self.mean = mean
        self.var = var
        self.std = np.sqrt(self.var)
        self.point = points
        self.data0 = np.random.normal(loc=self.mean, scale=self.std, size=self.point)
        self.data1 = np.random.normal(loc=self.mean, scale=self.std, size=self.point)
        self.data2 = np.random.normal(loc=self.mean, scale=self.std, size=self.point)
        self.data3 = np.random.normal(loc=self.mean, scale=self.std, size=self.point)
        self.data4 = np.random.normal(loc=self.mean, scale=self.std, size=self.point)
        self.data5 = np.random.normal(loc=self.mean, scale=self.std, size=self.point)
        self.data6 = np.random.normal(loc=self.mean, scale=self.std, size=self.point)
        self.data7 = np.random.normal(loc=self.mean, scale=self.std, size=self.point)
        self.data8 = np.random.normal(loc=self.mean, scale=self.std, size=self.point)
        self.data9 = np.random.normal(loc=self.mean, scale=self.std, size=self.point)

    def __str__(self):
        return "Data Generation for 10 sets of data"

    def sub_data(self, index=0):
        if index == 0:
            return self.data0
        elif index == 1:
            return self.data1
        elif index == 2:
            return self.data2
        elif index == 3:
            return self.data3
        elif index == 4:
            return self.data4
        elif index == 5:
            return self.data5
        elif index == 6:
            return self.data6
        elif index == 7:
            return self.data7
        elif index == 8:
            return self.data8
        elif index == 9:
            return self.data9
        else:
            print("index out of range")
            return self.data0


# Class Maximum Likelihood Estimate
class MaximumLikelihoodEstimate:
    def __init__(self, data):
        self.data = data
        self.data0 = self.data.data0

    def estimate_mean(self, data=None, data_points=None):
        # Determine number of points
        if data_points is None:
            data_points = self.data0.sub_data(index=0).shape[0]
        else:
            data_points = data_points
        # Determine Data
        if data is None:
            data = self.data.sub_data(index=0)
        else:
            data = data
        # Estimate the mean
        estimated_mean = 1 / data_points * np.sum(data[:data_points])
        # Calculate the error
        error = np.sqrt((ACTUAL_MEAN - estimated_mean) ** 2) * 100
        # Return error
        return error

    def error_plot(self, average=False, show=True):
        # For single set of data
        if not average:
            # Create Empty Storage
            point_num = []
            error_list = []
            # Calculate each estimated mean regarding number of points taken
            for index in range(0, int(np.log10(POINTS))):
                point_num.append(index)
                error_list.append(self.estimate_mean(data=None, data_points=10 ** index))
            # Generate the plot
            plt.plot(point_num, error_list, linewidth=1, color='r', linestyle='--', marker='o', label="Single Set")
            plt.xlabel("Number of Points [10^x]")
            plt.ylabel("Error Rate (%)")
            plt.title("Error in Estimate vs. Number of Points used in Estimate")
            plt.legend(loc=1)
            if show:
                plt.show()
            else:
                plt.savefig("error_vs_points_mle_single.png")
        # For multi sets of data
        else:
            point_num = []
            error_list_temp = []
            error_list = []
            for num in range(0, int(np.log10(POINTS))):
                # Generate error for 10 sets
                for index in range(0, SET_NUM):
                    err = self.estimate_mean(data=self.data.sub_data(index=index), data_points=10 ** num)
                    error_list_temp.append(err)
                # Calculate the average error
                error_list.append(sum(error_list_temp) / len(error_list_temp))
                point_num.append(num)
                error_list_temp = []
            plt.plot(point_num, error_list, linewidth=1, color='b', linestyle='--', marker='o', label="10 Sets Average")
            plt.xlabel("Number of Points [10^x]")
            plt.ylabel("Error Rate (%)")
            plt.title("Error in Estimate vs. Number of Points used in Estimate")
            plt.legend(loc=1)
            if show:
                plt.show()
            else:
                plt.savefig("error_vs_points_mle_multi.png")


class BayesianEstimate:
    def __init__(self, data, guessed_std=1):
        self.data = data
        self.guessed_std = guessed_std

    def estimate_mean(self, guessed_mean, data=None, data_points=None):
        # Determine number of points
        if data_points is None:
            data_points = self.data.sub_data(index=0).shape[0]
        else:
            data_points = data_points
        # Determine Data
        if data is None:
            data = self.data.sub_data(index=0)
        else:
            data = data
        # Estimate the mean
        weight_future = (data_points * self.data.var) / (data_points * self.data.var + self.guessed_std ** 2)
        weight_guess = self.guessed_std ** 2 / (data_points * self.data.var + self.guessed_std ** 2)
        estimated_mean = weight_future * np.mean(data[:data_points]) + weight_guess * guessed_mean
        # Error Calculation
        error = np.sqrt((ACTUAL_MEAN - estimated_mean) ** 2) * 100
        # Return Error
        return error

    def error_plot(self, guessed_mean, show=True):
        # Create Empty Storage
        point_num = []
        error_list = []
        # Calculate each estimated mean regarding number of points taken
        for index in range(0, int(np.log10(POINTS))):
            point_num.append(index)
            error_list.append(self.estimate_mean(guessed_mean=guessed_mean, data=None, data_points=10 ** index))
        # Generate the plot
        plt.plot(point_num, error_list, linewidth=1, color='g', linestyle='--', marker='o', label="Bayesian Single Set")
        plt.xlabel("Number of Points [10^x]")
        plt.ylabel("Error Rate (%)")
        plt.title("Error in Estimate vs. Number of Points used in Estimate")
        plt.legend(loc=1)
        if show:
            plt.show()
        else:
            plt.savefig("error_vs_points_be.png")

##############################################################
#   Function Prototype
##############################################################


##############################################################
#   Main Function
##############################################################
def main():
    print("Hello World!")
    data_set = DataGen()
    mle = MaximumLikelihoodEstimate(data=data_set)
    # Generate MLE plot for a single set
    mle.error_plot(average=False, show=False)
    # Generate MLE plot for average of 10 sets
    mle.error_plot(average=True, show=False)
    be = BayesianEstimate(data=data_set)
    be.error_plot(guessed_mean=2, show=False)


##############################################################
#    Main Function Runner
##############################################################
if __name__ == "__main__":
    main()
