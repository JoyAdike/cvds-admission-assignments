"""
exercise3.py : Debugging Exercise 3: CSV data type bug.

Bug found: csv.reader returns all values as strings, not numbers.
When numpy stacks these string values into an array, matplotlib treats them as text categories sorted alphabetically rather than as numerical values sorted mathematically.

This means the precision-recall curve is plotted in the wrong order : "0.962" comes before "1.0" alphabetically but after it numerically.

Fix: Convert the numpy array to float immediately after stacking, so all values are treated as numbers during plotting.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt


def plot_data(csv_file_path: str):
    """
    This code plots the precision-recall curve based on data from a .csv file, where precision is on the x-axis and recall is on the y-axis.

    :param csv_file_path: The CSV file containing the data to plot.

    Bug: csv.reader returns every value as a string. numpy.stack creates a string array. Matplotlib plots strings as categorical text values sorted alphabetically, not numerically — so the curve appears in the wrong order.

    Fix: Convert results to float after stacking so matplotlib treats all values as numbers.

    >>> f = open("data_file.csv", "w")
    >>> w = csv.writer(f)
    >>> _ = w.writerow(["precision", "recall"])
    >>> w.writerows([[0.013,0.951],[0.376,0.851],[1.0,0.0]])
    >>> f.close()
    >>> plot_data('data_file.csv')
    """
    # Load data from CSV
    results = []
    with open(csv_file_path) as result_csv:
        csv_reader = csv.reader(result_csv, delimiter=',')
        next(csv_reader)  # skip header row
        for row in csv_reader:
            results.append(row)

    # ORIGINAL BUGGY CODE:
    # results = np.stack(results)
    # Bug: np.stack on a list of string lists creates a string array
    # Matplotlib then sorts values alphabetically not numerically

    # FIX: convert to float immediately after stacking
    results = np.stack(results).astype(float)

    # Plot precision-recall curve
    # Column 0 = precision (x-axis), Column 1 = recall (y-axis)
    plt.plot(results[:, 0], results[:, 1])
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.show()


# Test the fix
if __name__ == "__main__":
    # Generate the test CSV file
    f = open("data_file.csv", "w")
    w = csv.writer(f)
    _ = w.writerow(["precision", "recall"])
    w.writerows([[0.013, 0.951],
                 [0.376, 0.851],
                 [0.441, 0.839],
                 [0.570, 0.758],
                 [0.635, 0.674],
                 [0.721, 0.604],
                 [0.837, 0.531],
                 [0.860, 0.453],
                 [0.962, 0.348],
                 [0.982, 0.273],
                 [1.0,   0.0]])
    f.close()

    print("Plotting precision-recall curve...")
    print("Expected: smooth curve going from top-left to bottom-right")
    plot_data('data_file.csv')
    print("Exercise 3 — Plot displayed correctly.")