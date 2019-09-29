import tensorflow as tf 

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

print('aaaaaaaaaaaasoftmax回归测试MNIST数据集正确率')

mnist = input_data.read_data_sets('./MNIST_data/',one_hot=True)
tf.logging.set_verbosity(old_v)

print('bbbbbbbbbbbbbsoftmax回归测试MNIST数据集正确率')

print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)

sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,784])#placeholder输入数据的地方 784=28*28维的向量

# variable用来存储模型参数 持久化    不同于tensor用后就消失
W=tf.Variable(tf.zeros([784,10]))#初始化为0
b=tf.Variable(tf.zeros([10]))

y=tf.nn.softmax(tf.matmul(x,W)+b)#预测label
y_=tf.placeholder(tf.float32,[None,10])#真实label

cross_entroy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))#交叉熵作为loss function
#reduce_mean对每个batch求均值  reduce_sum求和

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entroy)#得到进行训练的操作
#选择优化器 学习速率 优化目标（最小化cross-entry)

tf.global_variables_initializer().run()#全局参数初始化器  并执行他的run方法

for i in range(1000):#每次取100条（随机梯度下降）构成mini-batch并feed给placeholder,调用train_step训练
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})

#准确率
correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('softmax回归测试MNIST数据集正确率')

print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))
