import tensorflow as tf

state=tf.Variable(0,name="counter")
print(state.name)
one=tf.constant(1)

new_value=tf.add(state,one)
update=tf.assign(state,new_value)

init=tf.initialize_all_variables()

with tf.Session() as sess:
    #实例化变量之后，变量值会保存
    sess.run(init)
    for _ in range(3):
      print(sess.run(update))
      print(sess.run(state))