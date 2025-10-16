from __future__ import annotations
from typing import List, Optional


def binary_search(nums: List[int], target: int) -> int:
    """
    Return the index of `target` in sorted list `nums`, or -1 if not found.

    Example
    -------
    >>> binary_search([1, 3, 5, 7, 9], 7)
    3
    """
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def quicksort(arr: List[int]) -> List[int]:
    """
    In-place quicksort (returns a new sorted list for simplicity).

    Example
    -------
    >>> quicksort([3, 1, 4, 1, 5, 9])
    [1, 1, 3, 4, 5, 9]
    """
    if len(arr) <= 1:
        return arr[:]
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)


def two_sum(nums: List[int], target: int) -> Optional[tuple[int, int]]:
    """
    Return indices (i, j) such that nums[i] + nums[j] == target, or None.

    Example
    -------
    >>> two_sum([2, 7, 11, 15], 9)
    (0, 1)
    """
    seen = {}
    for i, x in enumerate(nums):
        comp = target - x
        if comp in seen:
            return (seen[comp], i)
        seen[x] = i
    return None
