import numpy as np

#pip --version
#pip install numpy

print(np.__version__)

a=[1,2,3]
b=[4,5,6]
result=[]
for i in range(len(a)):
    result.append(a[i]+b[i])
print(result)
#numpy way
a=np.array([1,2,3])
b=np.array([4,5,6])
result=a+b
print(result)
print(result.dtype) #data type of array

#multi dimentional array
a=np.array([[1,2,3],[4,5,6]])
print(a)

#
arr=np.ones((3,4)) #3 rows and 4 columns
print(arr)
print(arr.shape) #shape of array

arr1=np.arange(0,10,2) #start,stop,step
print(arr1)

arr2=np.arange(6).reshape(2,3) #reshape array
print(arr2)

a=np.array([1,2,3,4,5,6])
print(a[1:4]) #slicing array

print(a[1:4:2]) #slicing with step
print(a[:4]) #slicing from start to index 4
print(a[2:]) #slicing from index 2 to end
print(a[-3:]) #slicing last 3 elements
print(a.mean()) #mean of array
print(a.sum()) #sum of array
print(a.max()) #max of array
