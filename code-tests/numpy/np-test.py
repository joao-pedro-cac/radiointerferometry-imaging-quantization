import numpy as np

BARSPACE = 40

arr1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
arr2 = np.array([1, 2, 3, 4, 5], ndmin=3)              # ndmin sets the minimum number of dimensions
arr3 = np.array(["apple", "banana", "orange"])
arr4 = np.array([True, True, False, False, True])
arr5 = np.array([1, 2, 3, 4], dtype="S")               # all the data is converted into strings
arr5 = np.array([1, 2, 3, 4], dtype="i4")              # all the data is converted into 4-byte integers
arr6 = np.array([1.1, 2.2, 3.3])

print(arr1)          # array itself
print(type(arr1))    # type (ndarray)
print(arr1.ndim)     # dimensions

print("-" * BARSPACE)

print(arr2)          # array itself
print(type(arr2))    # type (ndarray)
print(arr2.ndim)     # dimensions

print("-" * BARSPACE)

print(arr1.dtype)     # data type

print("-" * BARSPACE)

print(arr3)
print(arr3.dtype)

print("-" * BARSPACE)

print(arr4)
print(arr4.dtype)

print("-" * BARSPACE)

print(arr5)
print(arr5.dtype)

print("-" * BARSPACE)

new_arr6 = arr6.astype('i1')

print(arr6)
print(arr6.dtype)

print(new_arr6)
print(new_arr6.dtype)

print("-" * BARSPACE)

arr6_copy = arr6.copy()
arr6_view = arr6.view()

arr6[0] = 5.5
arr6_copy[1] = 6.6
arr6_view[2] = 7.7

print(arr6)
print(arr6_copy)
print(arr6_view)

print(arr6.base)
print(arr6_copy.base)
print(arr6_view.base)

print("-" * BARSPACE)

arr_random = (5 * np.random.randn(5, 6)).astype('i1')
print(arr_random)
print(arr_random.shape)

print("-" * BARSPACE)

arr_random_reshaped = arr_random.reshape(3, 10)
print(arr_random_reshaped)
print()
print(arr_random_reshaped.reshape(10, 3))
print()
print(arr_random)
print()
print(arr_random_reshaped.reshape(5, 6))
print(arr_random_reshaped.base)