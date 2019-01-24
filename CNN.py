import scipy.io
import tensorflow as tf
import math
import numpy as np

import warnings
warnings.filterwarnings("ignore")

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model(features, labels, mode):
    width = height = int(math.sqrt(features['x'].get_shape().as_list()[1]))
    input_layer = tf.cast(tf.reshape(features['x'], [-1, width, height, 1]), tf.float16)  # Input layer
    if labels is not None:
        labels = tf.cast(labels, tf.int32)

    # Convolutional Layer 1  # Output size:(batch_size,28,28,32)
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)

    # Pooling Layer 1  # Output size:(batch_size,14,14,32)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

    # Convolutional Layer 2 # Output size:(batch_size,14,14,64)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)

    # Pooling Layer 2 # Output size:(batch_size,7,7,64)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

    # Dense layer 1 # Output size:(batch_size,1024)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # Dropout regularization
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense layer 2 # Output size:(batch_size,62)
    logits = tf.layers.dense(inputs=dropout, units=62)
    predictions = {"classes": tf.argmax(input=logits, axis=1),
                   "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}  # Probabilities from the logit layer
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)  # Loss function:Cross-entropy
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # global_step to count the number of training steps
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Accuracy metric
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def cnn_segmentation(features, labels, mode):
    input_layer = tf.cast(tf.reshape(features['x'], [-1, 28, 28, 1]), tf.float16)  # Input layer
    if labels is not None:
        labels = tf.cast(labels, tf.int32)

    # Convolutional Layer  # Output size:(batch_size,28,28,24)
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=24, kernel_size=5, padding="same", activation=tf.nn.relu)

    # Pooling Layer  # Output size:(batch_size,14,14,24)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

    # Dense layer 1 # Output size:(batch_size,120)
    pool1_flat = tf.reshape(pool1, [-1, 14 * 14 * 24])
    dense = tf.layers.dense(inputs=pool1_flat, units=120, activation=tf.nn.relu)
    # Dropout regularization
    dropout = tf.layers.dropout(inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense layer 2 # Output size:(batch_size,1)
    logits = tf.layers.dense(inputs=dropout, units=1)
    predictions = {"classes": tf.round(tf.nn.sigmoid(logits)),
                   "probabilities": tf.nn.sigmoid(logits, name="Sigmoid_tensor")}  # Probabilities from the logit layer
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)  # Loss function:Cross-entropy
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # global_step to count the number of training steps
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Accuracy metric
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused):
    directory = 'matlab/'
    X = scipy.io.loadmat(directory+'emnist-byclass')
    train_data = X['dataset'][0][0][0][0][0][0]
    train_labels = X['dataset'][0][0][0][0][0][1]
    eval_data = X['dataset'][0][0][1][0][0][0]
    eval_labels = X['dataset'][0][0][1][0][0][1]

    # CNN for character recognition
    classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir='tmp/convnet_model')

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=100,
                                                        shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    ###########################################################################################

    X = np.load('cropped_non_chars_imgs.npz')
    train_data = X['train']
    eval_data = X['test']
    train_labels = X['train_label']
    eval_labels = X['test_label']

    # CNN for character segmentation
    classifier = tf.estimator.Estimator(model_fn=cnn_segmentation, model_dir='tmp1/convnet_seg_model')

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "Sigmoid_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=100,
                                                        shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=3000, hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
    tf.app.run()
