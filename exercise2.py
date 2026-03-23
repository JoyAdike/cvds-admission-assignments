"""
exercise2.py : Debugging Exercise 2: NumPy coordinate swap bug.

Two bugs I found:

Bug 1: Wrong column index on the right side of the assignment.
    coords[:, 0], coords[:, 1] = coords[:, 1], coords[:, 1]
    The second value should be coords[:, 0], not coords[:, 1].
    Column 1 was being assigned to itself instead of the original column 0.

Bug 2 (Hidden): NumPy arrays are views into memory, not independent copies.
    Even after fixing Bug 1, simultaneous assignment in NumPy overwrites values in place. When coords[:, 0] = coords[:, 1] executes first, column 0 is immediately overwritten in memory. The subsequent coords[:, 1] = coords[:, 0] then reads the already-overwritten value, so column 1 never receives the original column 0 values.

Fix: Make explicit copies of both columns before swapping, so the original values are preserved independently before any assignment.

My thought process: To swap the contents of two glasses, you need a third empty glass as a temporary holder  Without it, you lose one of the original contents.
"""

import numpy as np


def swap(coords: np.ndarray) -> np.ndarray:
    """
    This method will flip the x and y coordinates in the coords array.

    :param coords: A numpy array of bounding box coordinates with
        shape [n,5] in format:
            [[x11, y11, x12, y12, classid1],
             [x21, y21, x22, y22, classid2],
             ...
             [xn1, yn1, xn2, yn2, classid3]]

    :return: The new numpy array where x and y coordinates are flipped.

    Bug 1: coords[:, 1] used twice on right side, should be coords[:, 0]
    Bug 2: NumPy views cause in-place overwrite during simultaneous assignment, original values lost before second assignment

    Fix: Copy both columns before performing any assignment.

    >>> import numpy as np
    >>> coords = np.array([[10, 5, 15, 6, 0],
    ...                    [11, 3, 13, 6, 0],
    ...                    [5, 3, 13, 6, 1]])
    >>> swapped = swap(coords)
    """
    # ORIGINAL CODE:
    # coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3] = \
    #     coords[:, 1], coords[:, 1], coords[:, 3], coords[:, 2]
    # Bug 1: second value is coords[:, 1] — should be coords[:, 0]
    # Bug 2: NumPy views overwrite memory during assignment

    # FIX: make explicit copies of all columns before swapping
    x1 = coords[:, 0].copy()  # save original x1
    y1 = coords[:, 1].copy()  # save original y1
    x2 = coords[:, 2].copy()  # save original x2
    y2 = coords[:, 3].copy()  # save original y2

    # Now swap x and y safely using the saved copies
    coords[:, 0] = y1  # x1 position gets original y1
    coords[:, 1] = x1  # y1 position gets original x1
    coords[:, 2] = y2  # x2 position gets original y2
    coords[:, 3] = x2  # y2 position gets original x2

    return coords


# Test the fix
if __name__ == "__main__":
    original = np.array([[10, 5, 15, 6, 0],
                         [11, 3, 13, 6, 0],
                         [5,  3, 13, 6, 1],
                         [4,  4, 13, 6, 1],
                         [6,  5, 13, 16, 1]])

    print("Original coordinates:")
    print(original)

    result = swap(original.copy())

    print("\nSwapped coordinates:")
    print(result)

    print("\nVerification — first row:")
    print(f"  Original: x1=10, y1=5, x2=15, y2=6")
    print(f"  Swapped:  x1={result[0,0]}, y1={result[0,1]}, "
          f"x2={result[0,2]}, y2={result[0,3]}")
    print(f"  Expected: x1=5,  y1=10, x2=6,  y2=15")

    assert result[0, 0] == 5,  "x1 should be original y1"
    assert result[0, 1] == 10, "y1 should be original x1"
    assert result[0, 2] == 6,  "x2 should be original y2"
    assert result[0, 3] == 15, "y2 should be original x2"
    assert result[0, 4] == 0,  "classid should be unchanged"

    print("\nExercise 2 — All checks passed.")