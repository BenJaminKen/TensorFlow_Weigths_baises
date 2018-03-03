import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    # random_normal: 正太分布随机数，均值mean,标准差stddev
    # truncated_normal:截断正态分布随机数，均值mean,标准差stddev,不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数
    # random_uniform:均匀分布随机数，范围为[minval,maxval]
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    #生成的数据所有未0.1 [[0.1,......,0.1]]
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    #tf.matmul()  为矩阵乘法
    #tf.multiply() 为矩阵点乘
    #np.dot() 为矩阵乘法
    #np.multiply() 为矩阵点乘
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs 
#在指定的间隔内返回均匀间隔的数字。
#返回num均匀分布的样本，在[start, stop]。

#np.newaxis的用法
#a=np.array([1,2,3,4,5])
#print a.shape

#print a

#输出结果

#(5,)
#[1 2 3 4 5]
#可以看出a是一个一维数组,
#x_data=np.linspace(-1,1,300)[:,np.newaxis]
#a=np.array([1,2,3,4,5])
#b=a[np.newaxis,:]
#print a.shape,b.shape
#print a
#print b
#输出结果：

#(5,) (1, 5)
#[1 2 3 4 5]
#[[1 2 3 4 5]]

#x_data=np.linspace(-1,1,300)[:,np.newaxis]
#a=np.array([1,2,3,4,5])
#b=a[:,np.newaxis]
#print a.shape,b.shape
#print a
#print b
#输出结果
#(5,) (5, 1)
#[1 2 3 4 5]
#[[1]
# [2]
# [3]
# [4]
# [5]]
#creat train data
x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_date=np.square(x_data)-0.5+noise
#end
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
prediciton=add_layer(l1,10,1,activation_function=None)
#按行求和reduction_indices=[1]
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediciton),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init=tf.global_variables_initializer()
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_date)
plt.ion()
plt.show()

sess= tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_date})
    if i%50==0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_date}))
        try:
           ax.lines.remove(lines[0])
        except Exception:
            pass
        prediciton_value=(sess.run(prediciton,feed_dict={xs:x_data}))
        lines=ax.plot(x_data,prediciton_value,'r-',lw=5)
        plt.pause(0.1)
sess.close()