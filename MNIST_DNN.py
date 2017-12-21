import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#PLACEHOLDER
x = tf.placeholder(tf.float32,shape=[None,784])

# VARIABLES
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#GRAPH
y = tf.matmul(x,W) + b

#LOSS FUNCTION
y_true = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))

#OPTIMIZER
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

#CREATE SESSION
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)


    for step in range(3000):

        batch_x, batch_y = mnist.train.next_batch(100)

        sess.run(train,feed_dict={x:batch_x,y_true:batch_y})

    #Evaluate Model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))
    #[TRUE, FALSE, FALSE, TRUE.....] --> [1,0,0,1.....]
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    #PREDICTED [3,4] TRUE [3,9]
    # [TRUE, FALSE]
    # [1,0]
    # 0.5 (Accuracy)

    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
