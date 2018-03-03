import tensorflow as tf
#1*2数组，二维数组
martix_1=tf.constant([[3,3]]);
#2*1数组，二维数组
martix_2=tf.constant([[2],[2]])
product=tf.matmul(martix_1,martix_2)
#method 1
sess=tf.Session()
result=sess.run(product)
print(result)
sess.close()

#mothod 2
with tf.Session() as sess:
    result=sess.run(product)
    print(result)