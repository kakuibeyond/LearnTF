'''
Created on 2019年8月6日

@author: chengwei

添加两个隐藏层
可视化过程

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    #定义成normal distribution（随机变量）要比初始化为全0更好
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

#make up some real data
x_data=np.linspace(-1,1,300)[:,np.newaxis]#[-1,1]区间，300个单位
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

xs=tf.placeholder(tf.float32,[None,1])#None表示任意个都可以
ys=tf.placeholder(tf.float32,[None,1])

l1=add_layer(xs,1, 10, activation_function=tf.nn.relu)
predition=add_layer(l1, 10, 1, activation_function=None)

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),
                   reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

fig=plt.figure()#图片框
ax=fig.add_subplot(1,1,1)#连续性的画图
ax.scatter(x_data,y_data)#真实数据 以点的形式plot上来
plt.ion()#show了之后主程序不暂停 继续往下
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
#         print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        #to see the step improvement
#         try:
#             ax.lines.remove(lines[0])
#         except Exception:
#             pass

        predition_value=sess.run(predition,feed_dict={xs:x_data})
        lines=ax.plot(x_data,predition_value,'r-',lw=5)
        plt.pause(1)#暂停1秒
        ax.lines.remove(lines[0])
    