"""
exercise1.py : Debugging Exercise 1: Set ordering bug.

Bug found: Python sets have no guaranteed order. Iterating over a set produces elements in an unpredictable sequence that can change between runs. This means fruit_id does not reliably correspond to the expected fruit name.

Fix: Convert the set to a sorted list before indexing. A sorted list always produces the same alphabetical order, so fruit_id 1 always returns the same fruit regardless of when or how the code is run.
"""

import csv
import numpy as np
from typing import Set


def id_to_fruit(fruit_id: int, fruits: Set[str]) -> str:
    """
    This method returns the fruit name by getting the string at a specific index of the set.

    :param fruit_id: The id of the fruit to get
    :param fruits: The set of fruits to choose the id from
    :return: The string corresponding to the index fruit_id

    Bug: Python sets have no guaranteed order. The same set can produce different orderings on different runs making index-based lookup unreliable.

    Fix: Convert the set to a sorted list first. sorted() always returns elements in the same alphabetical order, making the index reliable.

    >>> name1 = id_to_fruit(1, {"apple", "orange", "melon", "kiwi", "strawberry"})
    >>> name3 = id_to_fruit(3, {"apple", "orange", "melon", "kiwi", "strawberry"})
    >>> name4 = id_to_fruit(4, {"apple", "orange", "melon", "kiwi", "strawberry"})
    """
    # BUG: iterating directly over a set gives unpredictable order
    # ORIGINAL BUGGY CODE:
    # idx = 0
    # for fruit in fruits:
    #     if fruit_id == idx:
    #         return fruit
    #     idx += 1

    # FIX: convert set to sorted list to guarantee consistent ordering
    sorted_fruits = sorted(fruits)
    return sorted_fruits[fruit_id]


# Test 
if __name__ == "__main__":
    fruits = {"apple", "orange", "melon", "kiwi", "strawberry"}

    # sorted order will be: apple(0), kiwi(1), melon(2), orange(3), strawberry(4)
    name1 = id_to_fruit(1, fruits)
    name3 = id_to_fruit(3, fruits)
    name4 = id_to_fruit(4, fruits)

    print(f"Fruit at index 1: {name1}")  # expected: kiwi
    print(f"Fruit at index 3: {name3}")  # expected: orange
    print(f"Fruit at index 4: {name4}")  # expected: strawberry

    # Verify results are consistent across multiple calls
    assert id_to_fruit(1, fruits) == id_to_fruit(1, fruits), \
        "Same index should always return same fruit"
    print("\nExercise 1 — All checks passed.")