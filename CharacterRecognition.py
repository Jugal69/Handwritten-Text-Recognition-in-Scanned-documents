import tensorflow as tf
from CNN import cnn_model, cnn_segmentation
import numpy as np
import cv2
import scipy.io


def characterRecognizer(img):
    class_character = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'a',
                       11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j', 20: 'k',
                       21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't', 30: 'u',
                       31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z', 36: 'A', 37: 'B', 38: 'C', 39: 'D', 40: 'E',
                       41: 'F', 42: 'G', 43: 'H', 44: 'I', 45: 'J', 46: 'K', 47: 'L', 48: 'M', 49: 'N', 50: 'O',
                       51: 'P', 52: 'Q', 53: 'R', 54: 'S', 55: 'T', 56: 'U', 57: 'V', 58: 'W', 59: 'X', 60: 'Y',
                       61: 'Z'}
    width = 28
    X = (cv2.resize(img, (width, width))/255).reshape(1, width**2)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X}, y=None, shuffle=False)
    classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir='tmp/convnet_model')
    results = classifier.predict(input_fn=test_input_fn)
    for r in results:
        output = class_character[r['classes']]
    return output


def characterSegmentor(img1, img2):
    width = 28
    X1 = (cv2.resize(img1, (width, width)) / 255).reshape(1, width ** 2)
    X2 = (cv2.resize(img2, (width, width)) / 255).reshape(1, width ** 2)
    X = np.vstack((X1, X2))
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": X}, y=None, shuffle=False)
    classifier = tf.estimator.Estimator(model_fn=cnn_segmentation, model_dir='tmp1/convnet_seg_model')
    results = classifier.predict(input_fn=test_input_fn)
    output = [r for r in results]
    return True if output[0] == output[1] == 1 else False


directory1 = 'data/'
directory = 'matlab/'
X = scipy.io.loadmat(directory+'emnist-byclass')
data = X['dataset'][0][0][0][0][0][0]
labels = X['dataset'][0][0][0][0][0][1]

size = int(data.shape[0]*0.7)
train_data = data[:size]
train_labels = labels[:size]
eval_data = data[size:]
eval_labels = labels[size:]
test_data = X['dataset'][0][0][1][0][0][0]
test_labels = X['dataset'][0][0][1][0][0][1]

count = 0
for i in range(train_data.shape[0]):
    if train_labels[i][0] == characterRecognizer(train_data[i]):
        count += 1
print('Train accuracy:', count*100/train_data.shape[0])

count = 0
for i in range(eval_data.shape[0]):
    if eval_labels[i][0] == characterRecognizer(eval_data[i]):
        count += 1
print('Evaluation accuracy:', count*100/eval_data.shape[0])

count = 0
for i in range(test_data.shape[0]):
    if test_labels[i][0] == characterRecognizer(test_data[i]):
        count += 1
print('Test accuracy:', count*100/test_data.shape[0])

data = np.load('cropped_non_chars_imgs.npz')
size = int(data['train'].shape[0]*0.7)
train_data = data['train'][:size]
train_labels = data['train_label'][:size]
eval_data = data['train'][size:]
eval_labels = data['train_label'][size:]

test_data = data['test']
test_labels = data['test_label']

test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": test_data}, y=None, shuffle=False)
classifier = tf.estimator.Estimator(model_fn=cnn_segmentation, model_dir='tmp1/convnet_seg_model')
results = classifier.predict(input_fn=test_input_fn)
output = np.array([r['classes'] for r in results])
print(np.sum(output == test_labels)*100/test_data.shape[0])
