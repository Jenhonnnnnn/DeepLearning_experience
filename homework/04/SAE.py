# import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# Add the import statements
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

# # Visualize decoder setting
# # Parameters
# learning_rate = 0.01
# batch_size = 256
# display_step = 1
# examples_to_show = 10
#
# # Network Parameters
# n_input = 784  # 28x28 pix，即 784 Features
#
# # tf Graph input (only pictures)
# X = tf.placeholder("float", [None, n_input])
#
# # hidden layer settings
# n_hidden_1 = 256  # 经过第一个隐藏层压缩至256个
# n_hidden_2 = 128  # 经过第二个压缩至128个
# # 两个隐藏层的 weights 和 biases 的定义
# weights = {
#     'encoder_h1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
#     'encoder_h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
#     'decoder_h1': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_1])),
#     'decoder_h2': tf.Variable(tf.random.normal([n_hidden_1, n_input])),
# }
# biases = {
#     'encoder_b1': tf.Variable(tf.random.normal([n_hidden_1])),
#     'encoder_b2': tf.Variable(tf.random.normal([n_hidden_2])),
#     'decoder_b1': tf.Variable(tf.random.normal([n_hidden_1])),
#     'decoder_b2': tf.Variable(tf.random.normal([n_input])),
# }
#
#
# # Building the encoder
# def encoder(x):
#     # Encoder Hidden layer 使用的 Activation function 是 sigmoid #1
#     layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
#                                    biases['encoder_b1']))
#     # Decoder Hidden layer with sigmoid activation #2
#     layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
#                                    biases['encoder_b2']))
#     return layer_2
#
#
# # Building the decoder
# def decoder(x):
#     # Encoder Hidden layer with sigmoid activation #1
#     layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
#                                    biases['decoder_b1']))
#     # Decoder Hidden layer with sigmoid activation #2
#     layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
#                                    biases['decoder_b2']))
#     return layer_2


# '''
# Visualize encoder setting
# 只显示解压后的数据
learning_rate = 0.01    # 0.01 this learning rate will be better! Tested
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 20
# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
# hidden layer settings
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2  #将原有784Features 的数据压缩成2 Features数据
weights = {
    'encoder_h1': tf.Variable(tf.random.truncated_normal([n_input, n_hidden_1],)),
    'encoder_h2': tf.Variable(tf.random.truncated_normal([n_hidden_1, n_hidden_2],)),
    'encoder_h3': tf.Variable(tf.random.truncated_normal([n_hidden_2, n_hidden_3],)),
    'encoder_h4': tf.Variable(tf.random.truncated_normal([n_hidden_3, n_hidden_4],)),
    'decoder_h1': tf.Variable(tf.random.truncated_normal([n_hidden_4, n_hidden_3],)),
    'decoder_h2': tf.Variable(tf.random.truncated_normal([n_hidden_3, n_hidden_2],)),
    'decoder_h3': tf.Variable(tf.random.truncated_normal([n_hidden_2, n_hidden_1],)),
    'decoder_h4': tf.Variable(tf.random.truncated_normal([n_hidden_1, n_input],)),
}
biases = {
    'encoder_b1': tf.Variable(tf.random.normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random.normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random.normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random.normal([n_hidden_4])),
    'decoder_b1': tf.Variable(tf.random.normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random.normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random.normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random.normal([n_input])),#注意：在第四层时，输出量不再是 [0,1] 范围内的数，
    #而是将数据通过默认的 Linear activation function 调整为 (-∞,∞)
}
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                    biases['encoder_b4'])
    return layer_4
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                biases['decoder_b4']))
    return layer_4
# '''

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
# 比较原始数据与还原后的拥有 784 Features 的数据进行 cost 的对比，
# 根据 cost 来提升我的 Autoencoder 的准确率
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))  # 进行最小二乘法的计算(y_true - y_pred)^2
# loss = tf.reduce_mean(tf.square(y_true - y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Launch the graph
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(mnist.train.num_examples / batch_size)
    training_epochs = 20
    # Training cycle
    for epoch in range(training_epochs):  # 到好的的效果，我们应进行10 ~ 20个 Epoch 的训练
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, loss], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 20, figsize=(20, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))

    plt.show()

    # encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    # sc = plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels) #散点图
    # plt.colorbar(sc) #scatter设置颜色渐变条colorbar
    # plt.show()