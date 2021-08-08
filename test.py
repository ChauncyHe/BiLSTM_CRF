import tensorflow as tf
import numpy as np
a = np.arange(1,9).reshape([2,2,2])
print("第一片:",a[0],sep="\n",)
print("第二片:",a[1],sep="\n")
print("合并前两维后得到的矩阵:","\n",a.reshape([-1,2]))
b = np.eye(2,2)
print(a.reshape([-1,2]) @ b)
print(a[0] @ b)
print(a[1] @ b)