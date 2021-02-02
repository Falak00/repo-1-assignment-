#!/usr/bin/env python
# coding: utf-8

# # Assignment For Numpy
#    Difficulty Level <b>Beginner<b>
# 
#  1. Import the numpy package under the name np
# 

# In[1]:


import numpy as np


# #### 2.Create a null vector of size 10

# In[2]:


np.zeros(10)


# ####     3.Create a vector with values ranging from 10 to 49

# In[3]:


arr1 = np.arange(10,49)
arr1


# ####  4.Find the shape of previous array in question 3

# In[4]:


np.shape(arr1)


# #### 5.Print the type of the previous array in question 3

# In[8]:


arr = np.arange(10,49)
arr.dtype


# #### 6.Print the numpy version and the configuration

# In[9]:


np.version.version


# In[10]:


np.show_config()


# #### 7.Print the dimension of the array in question 3

# In[11]:


arr1 = np.arange(10,50)
arr1.ndim


# #### 8.Create a boolean array with all the True values

# In[12]:


bool_arr = np.ones(5, dtype = bool)
bool_arr


# #### 9.Create a two dimensional array

# In[13]:


arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_2d


# #### 10.Create a three dimensional array

# In[ ]:


arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
arr_3d


# # Difficulty Level <b>Easy<b>
# 
# #### 11.Reverse a vector (first element becomes last)

# In[14]:


arr1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
np.flip(arr1)


# #### 12.Create a null vector of size 10 but the fifth value which is 1

# In[15]:


arr = np.zeros(10) 
arr[4] = 1
arr


# #### 13.Create a 3x3 identity matrix

# In[16]:


arr = np.identity(3)
arr


# #### 14.arr = np.array([1, 2, 3, 4, 5])
# #### Convert the data type of the given array from int to float

# In[17]:


arr = np.array([1, 2, 3, 4, 5], dtype = float)
arr


# #### 15.
# arr1 = np.array([[1., 2., 3.],
# 
#             [4., 5., 6.]])  
# 
# arr2 = np.array([[0., 4., 1.],
# 
#            [7., 2., 12.]])
# Multiply arr1 with arr2

# In[18]:


arr1 = np.array([[1., 2., 3.],

        [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],

       [7., 2., 12.]])
arr = arr1 * arr2
arr


# #### 16.
# 
# arr1 = np.array([[1., 2., 3.],
# 
#             [4., 5., 6.]]) 
# 
# arr2 = np.array([[0., 4., 1.],
# 
#             [7., 2., 12.]])
# Make an array by comparing both the arrays provided above

# In[1]:


import numpy as np
arr1 = np.array([[1., 2., 3.] , [4., 5., 6.]])
arr2 = np.array([[0., 4., 1.] , [7., 2., 12.]])
arr = np.concatenate((arr1, arr2))
arr


# #### 17.Extract all odd numbers from arr with values(0-9)

# In[2]:


a = np.arange(9)
a[a % 2 == 1]


# #### 18.Replace all odd numbers to -1 from previous array

# In[3]:


a = np.arange(9)
a[a%2 != ]


# #### 19. Replace the values of indexes 5,6,7 and 8 to 12

# In[4]:


arr = np.arange(10)
arr[5:9] = 12
arr


# #### 20.Create a 2d array with 1 on the border and 0 inside

# In[7]:


x = np.ones((5,5))
print("Original array:")
print(x)
print("1 on the border and 0 inside in the array")
x[1:-1,1:-1] = 0
print(x)
print("An other way of this")
a = np.ones((3,3), dtype = "int32")
a[1,1] = 0
a


# # Difficulty Level Medium
# #### 21.arr2d = np.array([[1, 2, 3],
# 
#             [4, 5, 6], 
# 
#             [7, 8, 9]])
# #### Replace the value 5 to 12

# In[8]:


arr2d = np.array([[1, 2, 3],
                  [4, 5, 6], 
                  [7, 8, 9]])
arr2d[1:2,1:2] = 12
arr2d   


# #### 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# #### Convert all the values of 1st array to 64

# In[9]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]],
                  [[7, 8, 9], [10, 11, 12]]])

arr3d[0,:] = 64
arr3d


# #### 23.Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[10]:


arr2 = np.arange(10).reshape(2,5)
print(arr2)
arr2[0]


# #### 24.Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[12]:


arr2 = np.arange(10).reshape(2,5)
print(arr2)
arr2[1,1]


# #### 25.Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[13]:


arr2 = np.arange(10).reshape(2,5)
print(arr2)
arr2[0:2,:2]


# #### 26.Create a 10x10 array with random values and find the minimum and maximum values

# In[14]:


x = np.random.random((10,10))
print("Original Array:")
print(x) 
xmin, xmax = x.min(), x.max()
print("Minimum and Maximum Values:")
print(xmin, xmax)


# #### 27.a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# #### Find the common items between a and b

# In[17]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d([1,2,3,2,3,4,3,4,5,6],[7,2,10,2,7,4,9,4,9,8])


# #### 28.a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# #### Find the positions where elements of a and b match

# In[ ]:





# #### 29.names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# #### Find all the values from array data where the values from array names are not equal to Will

# In[18]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
data[names != "Will"]


# #### 30.names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# #### Find all the values from array data where the values from array names are not equal to Will and Joe

# In[19]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
mask = (names == 'Bob') | (names == 'Will')
mask
data[mask]


# 
# # Difficulty Level Hard
# 
# #### 31.Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[20]:


arr2d = np.arange(1.,16.).reshape(5,3)
arr2d


# #### 32.Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[21]:


arr3d = np.arange(1.,17.).reshape(2,2,4)
arr3d


# #### 33.Swap axes of the array you created in Question 32

# In[23]:


arr3d = np.arange(1.,17.).reshape(2,4,2)
arr3d


# #### 34.Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[24]:


arr = np.arange(10)
print(arr)
np.sqrt(arr)


# #### 35.Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[25]:


arr1 = np.random.randn(12)
arr2 = np.random.randn(12)
np.maximum(arr1,arr2)


# #### 36.names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# #### Find the unique names and sort them out!

# In[26]:


import numpy as np
x = np.array([10, 10, 20, 20, 30, 30])
print("Original array:")
print(x)
print("Unique elements of the above array:")
print(np.unique(x))


# #### 37.a = np.array([1,2,3,4,5]) b = np.array([5,6,7,8,9])
# #### From array a remove all items present in array b

# In[27]:


a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 6, 7, 8, 9])

result = np.setdiff1d(a, b)
print(result)


# 
# #### 38.Following is the input NumPy array delete column two and insert following new column in its place.
# #### sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]])
# 
# #### newColumn = numpy.array([[10,10,10]])

# In[28]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newCol = ([10,10,10])
sampleArray[:,2] = newCol
sampleArray


# #### 39.x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# #### Find the dot product of the above two matrix

# In[29]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y)


# #### 40. Generate a matrix of 20 random values and find its cumulative sum

# In[30]:


arr = np.random.random(20)
arr.cumsum()


# In[ ]:




