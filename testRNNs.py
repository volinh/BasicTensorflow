import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
from twisted.web.html import output

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

hm_epochs = 5
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128


x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step")

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes],name="weights")),
             'biases':tf.Variable(tf.random_normal([n_classes]),name="bias")}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    print(len(outputs))
    print(outputs[0].shape)
    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y),name="lost")
    optimizer = tf.train.AdamOptimizer().minimize(cost,global_step=global_step)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('checkpoint/skip-gram/skip-gram-2145')
        if ckpt and ckpt.model_checkpoint_path:
            print("restore ...")
            saver.restore(sess, ckpt.model_checkpoint_path)

        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for index in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            saver.save(sess, 'checkpoint/skip-gram/skip-gram', global_step=global_step)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',
              accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))


train_neural_network(x)


