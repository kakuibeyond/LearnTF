'''
Created on 2019年8月6日

@author: chengwei
'''
import tensorflow as tf

matrix1=tf.constant([[3,3]])#一行两列矩阵
matrix2=tf.constant([[2],   #两行一列的矩阵
                     [2]])
product=tf.matmul(matrix1,matrix2)#matrix multiply

#method 1
sess=tf.Session()
result=sess.run(product)
print(result)
sess.close()

#method 2
with tf.Session() as sess:#自动关闭
    result2=sess.run(product)
    print(result2)