'''
Created on 2019年8月6日

@author: chengwei

placeholder传入值
先hold住一个变量 但其值需要运行结果时外界传入

'''
import tensorflow as tf

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[5.],input2:[9.]}))