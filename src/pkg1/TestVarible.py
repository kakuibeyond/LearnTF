'''
Created on 2019年8月6日

@author: chengwei
'''
import tensorflow as tf

state=tf.Variable(1,name='counter')#变量
# print(state.name)#counter:0 0代表第一个变量
one=tf.constant(1)

new_value=tf.add(state,one)
update=tf.compat.v1.assign(state,new_value)#new_value的值赋值给state

init=tf.compat.v1.global_variables_initializer()#must have if define variable

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
