import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
digits=load_digits()
X=digits.data
y=digits.target
y=LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)

def add_layer(inputs,in_size,out_size,layer_name,activation_function=None):
    with tf.name_scope(layer_name+"/Weights"):
        Weights=tf.Variable(tf.random_normal([in_size,out_size]))
        tf.summary.histogram(layer_name+"/W",Weights)
    with tf.name_scope(layer_name+"/biases"):
        biases=tf.Variable(tf.zeros([1,out_size])+0.1)
        tf.summary.histogram(layer_name+"/b",biases)
    with tf.name_scope(layer_name+"/Wx_plus_b"):
        Wx_plus_b=tf.matmul(inputs,Weights)+biases
        Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob)
        tf.summary.histogram(layer_name+"/Wx+b",Wx_plus_b)
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

keep_prob=tf.placeholder(tf.float32)
xs=tf.placeholder(tf.float32,[None,64])
ys=tf.placeholder(tf.float32,[None,10])

l1=add_layer(xs,64,50,layer_name="l1",activation_function=tf.nn.tanh)
prediction=add_layer(l1,50,10,layer_name="l2",activation_function=tf.nn.softmax)
cross_entroy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar("loss",cross_entroy)
tran_step=tf.train.GradientDescentOptimizer(0.2).minimize(cross_entroy)
sess=tf.Session()
init=tf.global_variables_initializer()
merge=tf.summary.merge_all()
write_train=tf.summary.FileWriter("logs/train",sess.graph)
write_test=tf.summary.FileWriter("logs/test",sess.graph)
sess.run(init)
for i in range(1000):
    sess.run(tran_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.5})
    if i%50==0:
        train_result=sess.run(merge,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
        write_train.add_summary(train_result,i)
        test_train=sess.run(merge,feed_dict={xs:X_test,ys:y_test,keep_prob:1})
        write_test.add_summary(test_train,i)
