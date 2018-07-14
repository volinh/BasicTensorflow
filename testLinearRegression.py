import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
DATA_FILE = "data/slr05.xls"
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")

Y_predicted = X * w + b

loss = tf.square(Y - Y_predicted, name="loss")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):  # run 100 epochs
        for x, y in data:
            _,cost = sess.run([optimizer,loss], feed_dict={X: x, Y: y})
            print("loss : ", cost)
    print(w.eval())
    print(b.eval())

tf.train.GradientDescentOptimizer(0.01).apply_gradients()