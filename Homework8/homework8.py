# ECE 4110 - Intro to Machine Learning and Pattern Recognition
# The combined effort of Brandon Begaj, Casey Bruno, and Kuang Jiang
# Demo code for Dynamic Programming of string detection for sentences

##############################################################
#   Libraries
##############################################################
import numpy as np


##############################################################
#   Variable Definition
##############################################################
REWARD = 2
COST = -1


##############################################################
#   Class Definition
##############################################################
# Class of Dynamic Programming for String "Matching"
class StringMatching:
    # Initialize the reference and hypothesis
    def __init__(self, reference, hypothesis):
        # Store the string char by char
        self.reference = list(reference)
        self.hypothesis = list(hypothesis)
        # Create 2D matrix with coordinates
        self.match_matrix = np.zeros([len(self.hypothesis)+1, len(self.reference)+1])
        for index in range(0, len(self.hypothesis)+1):
            self.match_matrix[index, 0] = -index
        for index in range(0, len(self.reference)+1):
            self.match_matrix[0, index] = -index
        # Matching Process
        for row in range(1, len(self.hypothesis)+1):
            for column in range(1, len(self.reference)+1):
                self.match_matrix[row, column] = self.optimum_calculation(row=row, column=column)
        # Matching Path of the cell
        self.match_pass = np.array([len(self.hypothesis), len(self.reference)])
        row = len(self.hypothesis)
        column = len(self.reference)
        while row > 0 or column > 0:
            self.match_pass = np.row_stack([self.optimum_path(row, column),
                                            self.match_pass])
            row = self.match_pass[0, 0]
            column = self.match_pass[0, 1]

    # Find the optimum path via calculation
    def optimum_calculation(self, row, column):
        # Set up variable
        up = self.match_matrix[row - 1, column]
        left = self.match_matrix[row, column - 1]
        up_left = self.match_matrix[row - 1, column - 1]
        reference = self.reference[column - 1]
        hypothesis = self.hypothesis[row - 1]
        # Calculate Cost
        diagonal = up_left + REWARD if reference == hypothesis else up_left + COST
        vertical = up + COST
        horizontal = left + COST
        return max(diagonal, vertical, horizontal)

    # Return the result of the calculation
    def cost_result(self):
        return self.match_matrix[self.match_matrix.shape[0]-1, self.match_matrix.shape[1]-1]

    # Return the path result
    def path_result(self):
        return self.match_pass

    # Print the string path or return result
    def string_result(self, print_result=True, return_reference=False, return_hypothesis=False):
        reference_string = ""
        hypothesis_string = ""
        for index in range(1, self.match_pass.shape[0]):
            # construct hypothesis
            if self.match_pass[index, 0] == self.match_pass[index-1, 0]:
                hypothesis_string += "*"
            else:
                hypothesis_string += self.hypothesis[self.match_pass[index, 0] - 1].upper()
            # construct reference
            if self.match_pass[index, 1] == self.match_pass[index-1, 1]:
                reference_string += "*"
            else:
                reference_string += self.reference[self.match_pass[index, 1] - 1]
        # print both string
        if print_result:
            print(reference_string)
            print(hypothesis_string)
        # Return Value
        if return_reference:
            return return_reference
        if return_hypothesis:
            return return_hypothesis

    # Find the optimum path
    def optimum_path(self, row, column):
        diagonal = self.match_matrix[row-1, column-1]
        vertical = self.match_matrix[row-1, column]
        horizontal = self.match_matrix[row, column-1]
        previous_step = max(diagonal, vertical, horizontal)
        if previous_step == diagonal:
            return [row-1, column-1]
        elif previous_step == vertical:
            return [row-1, column]
        else:
            return [row, column-1]


# Class of Dynamic Programming for Sentence "Matching"
class SentenceMatching:
    # Initialize the reference and hypothesis
    def __init__(self, reference, hypothesis):
        # Separate Sentence to individual String
        self.reference = reference.lower().split(" ")
        self.hypothesis = hypothesis.lower().split(" ")
        # Create 2D matrix with coordinates
        self.match_matrix = np.zeros([len(self.hypothesis) + 1, len(self.reference) + 1])
        # Create 2D matrix with coordinates
        self.match_matrix = np.zeros([len(self.hypothesis) + 1, len(self.reference) + 1])
        # Create the first column for hypothesis
        for index in range(1, len(self.hypothesis) + 1):
            self.match_matrix[index, 0] = self.match_matrix[index-1, 0] + \
                                          self.first_error(word=self.hypothesis[index-1])
        # Create the first row for reference
        for index in range(1, len(self.reference) + 1):
            self.match_matrix[0, index] = self.match_matrix[0, index-1] + \
                                          self.first_error(word=self.reference[index-1])
        # Matching Process
        for row in range(1, len(self.hypothesis)+1):
            for column in range(1, len(self.reference)+1):
                self.match_matrix[row, column] = self.optimum_calculation(column=column, row=row)
        # Matching Pass to Cell
        self.match_pass = np.array([len(self.hypothesis), len(self.reference)])
        row = len(self.hypothesis)
        column = len(self.reference)
        while row > 0 or column > 0:
            self.match_pass = np.row_stack([self.optimum_path(row, column),
                                            self.match_pass])
            row = self.match_pass[0, 0]
            column = self.match_pass[0, 1]

    # Find the optimum path via calculation
    def optimum_calculation(self, column, row):
        # Determine values
        up = self.match_matrix[row - 1, column]
        left = self.match_matrix[row, column - 1]
        up_left = self.match_matrix[row - 1, column - 1]
        reference = self.reference[column - 1]
        hypothesis = self.hypothesis[row - 1]
        # Determine Reward and Cost multiplier
        multiplier_vertical = len(self.hypothesis[row - 1])
        multiplier_horizontal = len(self.reference[column - 1])
        multiplier_diagonal = len(self.reference[column - 1])
        # Calculate the Pass
        diagonal = up_left + REWARD * multiplier_diagonal if reference == hypothesis \
            else up_left + COST * multiplier_diagonal
        vertical = up + COST * multiplier_vertical
        horizontal = left + COST * multiplier_horizontal
        return max(diagonal, vertical, horizontal)

    # Find the optimum path
    def optimum_path(self, row, column):
        diagonal = self.match_matrix[row - 1, column - 1]
        vertical = self.match_matrix[row - 1, column]
        horizontal = self.match_matrix[row, column - 1]
        previous_step = max(diagonal, vertical, horizontal)
        if previous_step == diagonal:
            return [row - 1, column - 1]
        elif previous_step == horizontal:
            return [row, column - 1]
        else:
            return [row - 1, column]

    # Calculate the cost with empty words
    def first_error(self, word):
        reference = list(word)
        hypothesis = "*" * len(reference)
        return StringMatching(reference=reference, hypothesis=hypothesis).cost_result()

    # Return the result of the calculation
    def cost_result(self):
        return self.match_matrix[self.match_matrix.shape[0] - 1, self.match_matrix.shape[1] - 1]

    # Return the path result
    def path_result(self):
        return self.match_pass

    # Print String result
    def string_result(self, word_error=True):
        # String Setup
        reference_string = ""
        hypothesis_string = ""
        # Counter Setup
        insert = 0
        delete = 0
        substitute = 0
        # String Generation
        for index in range(1, self.match_pass.shape[0]):
            # construct hypothesis
            if self.match_pass[index, 0] == self.match_pass[index - 1, 0]:
                hypothesis_string += "-" * len(self.reference[self.match_pass[index, 1] - 1])
                delete += 1
            else:
                hypothesis_string += self.hypothesis[self.match_pass[index, 0] - 1]
            # construct reference
            if self.match_pass[index, 1] == self.match_pass[index - 1, 1]:
                reference_string += "-" * len(self.hypothesis[self.match_pass[index, 0] - 1])
                insert += 1
            else:
                reference_string += self.reference[self.match_pass[index, 1] - 1]
            hypothesis_string += " "
            reference_string += " "
        # Modification Preparation
        reference_string = reference_string.split(" ")
        hypothesis_string = hypothesis_string.split(" ")
        printable_reference = ""
        printable_hypothesis = ""
        # String Modification
        for index in range(0, len(reference_string)):
            # Count Controller Setup
            sub = False
            # Modification - if match
            if reference_string[index] == hypothesis_string[index]:
                printable_reference += reference_string[index]
                printable_hypothesis += hypothesis_string[index]
            # Modification - if not match but in the same str length
            elif len(reference_string[index]) == len(hypothesis_string[index]):
                if reference_string[index].isalpha():
                    printable_reference += reference_string[index].upper()
                    sub = True
                else:
                    printable_reference += reference_string[index]
                if hypothesis_string[index].isalpha():
                    printable_hypothesis += hypothesis_string[index].upper()
                    if sub:
                        substitute += 1
                else:
                    printable_hypothesis += hypothesis_string[index]
            else:
                size_fix = max(len(reference_string[index]), len(hypothesis_string[index]))
                printable_reference += reference_string[index].upper().rjust(size_fix)
                printable_hypothesis += hypothesis_string[index].upper().rjust(size_fix)
                substitute += 1
            printable_hypothesis += " "
            printable_reference += " "
        # Print both striPng
        print("Reference :", printable_reference)
        print("Hypothesis:", printable_hypothesis)
        # Print Error Result
        if word_error:
            print("Substitute:", substitute,
                  "Insert:", insert,
                  "Delete:", delete,
                  "Total Error:", substitute + insert + delete)


##############################################################
#   Function Prototype
##############################################################
# Test String Matching Class with input string
def string_match_test():
    print("Testing String Match Class")
    # Ask for input from the user
    string1 = str(input("Enter the reference string: ")).strip()
    string2 = str(input("Enter the hypothesis string: ")).strip()
    # Check if sentence is not empty
    if not string1 or not string2:
        print("Input Error: Empty String")
    else:
        string_match = StringMatching(reference=string1, hypothesis=string2)
        string_match.string_result()
        print("Cost result: ", string_match.cost_result())


# Test Sentence Matching Class with input sentence
def sentence_match_test():
    print("Testing Sentence Match Case")
    # Ask for input from the user
    sentence1 = str(input("Enter the reference sentence: ")).strip()
    sentence2 = str(input("Enter the hypothesis sentence: ")).strip()
    # Check if sentence is not empty
    if not sentence1 or not sentence2:
        print("Input Error: Empty String")
    else:
        string_match = SentenceMatching(reference=sentence1, hypothesis=sentence2)
        string_match.string_result()
        print("Cost result (character level): ", string_match.cost_result())


# Runner Function
def athletic():
    # Open and read two lines of the file
    reference = []
    hypothesis = []
    with open("string.txt", "r") as f:
        line_num = 0
        for line in f.readlines():
            if line_num % 2 == 0:
                reference.append(line.strip())
            if line_num % 2 == 1:
                hypothesis.append(line.strip())
            line_num += 1
    # Set Class
    for item_index in range(0, len(reference)):
        string_detection = SentenceMatching(reference=reference[item_index], hypothesis=hypothesis[item_index])
        # Print result
        string_detection.string_result()
        print()


##############################################################
#   Main Function
##############################################################
def main():
    print("Initiating ... \n")
    athletic()
    # string_match_test()
    # sentence_match_test()


##############################################################
#    Main Function Runner
##############################################################
if __name__ == "__main__":
    main()
