import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def compute_accuarcy(v_xs,v_ys):
    global prediction
    y_prediction=sess.run(prediction,feed_dict={xs:v_xs,ys:v_ys})
    correct_prediction=tf.equal(tf.argmax(y_prediction,1),tf.arg_max(v_ys,1))
    accuarcy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuarcy,feed_dict={xs:v_xs,ys:v_ys})
    return result


def add_layer(inputs,in_size,out_size,activefunction=None):
    with tf.name_scope("Weights"):
        Weights=tf.Variable(tf.random_normal([in_size,out_size]),name="W")
        tf.summary.histogram("W",Weights)
    with tf.name_scope("biases"):
        biases=tf.Variable(tf.zeros([1,out_size])+0.1,name="b")
        tf.summary.histogram("b",biases)
    with tf.name_scope("Wx_plus_b"):
        Wx_plus_b=tf.matmul(inputs,Weights)+biases
        tf.summary.histogram("w*x+b",Wx_plus_b)
    if(activefunction==None):
        outputs=Wx_plus_b
        return outputs
    else:
        outputs=activefunction(Wx_plus_b)
    tf.summary.histogram("outputs",outputs)
    return outputs

#define placeholder for inputs to network
xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])


#add output layer
prediction=add_layer(xs,784,10,tf.nn.softmax)
#tf.summary.scalar("prdeiction",prediction)
#the error between prediction and real data
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar("cross_entropy",cross_entropy)
tanin_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#tf.summary.scalar("tanin_step",tanin_step)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
merge=tf.summary.merge_all()
writer=tf.summary.FileWriter("logs/",sess.graph)
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(tanin_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%100==0:
        print(compute_accuarcy(mnist.test.images,mnist.test.labels))
        result=(sess.run(merge,feed_dict={xs:batch_xs,ys:batch_ys}))
        writer.add_summary(result,i)
writer.close()
sess.close()
