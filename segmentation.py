import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import cv2
import os
import shutil
from CharacterRecognition import characterRecognizer, characterSegmentor


def makePartitions(data1, data2):
    flag = f = False
    space = []
    count = 0
    for i in range(data1.shape[1]):
        if np.count_nonzero(data1[:, i]):
            if flag:
                space[-1][1] = count
            flag = False
            count = 0
            X = data2[:, i].reshape((data2.shape[0], 1)) if not f else np.hstack(
                (X, data2[:, i].reshape((data2.shape[0], 1))))
            f = True
        else:
            count += 1
            if not flag and not f:
                space.append([0, 0])
            elif not flag:
                space.append([X.shape[1], 0])
            flag = True
    return space, X


def isValid(index, width, maxWidth):
    return False if (index-width) < 0 or (index+width) > maxWidth else True


def displayImage(X):
    cv2.namedWindow('character', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('character', 500, 500)
    cv2.imshow('character', X)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def binarization(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = 0 if X[i, j] < 128 else X[i, j]
    return X


directory = 'data/'
data = abs(np.mean(cv2.imread(directory+'/a01-000u-00.png'), axis=2)-255)
data1 = binarization(data)
space, X = makePartitions(data1, data)
space_mat = np.array(space)
space_mat = list(filter(lambda a: a != 0, space_mat[space_mat[:, 1] > np.mean(space_mat, axis=0)[1], 0]))
count = 0
directoryWords = 'data/blocks/'
try:
    os.mkdir(directoryWords)
except FileExistsError:
    shutil.rmtree(directoryWords)
    os.mkdir(directoryWords)
for i in range(len(space_mat)):
    img = Image.fromarray(abs(X[:, count:space_mat[i]]).astype('uint8'))
    img.save(directoryWords+'temp'+str(i)+'.png')
    count = space_mat[i]
img = Image.fromarray(abs(X[:, count:]).astype('uint8'))
img.save(directoryWords+'temp'+str(i+1)+'.png')

output = ''
for j in range(len(space_mat)+1):
    X = np.mean(cv2.imread(directoryWords+'temp'+str(j)+'.png'), axis=2)
    space1, X1 = makePartitions(skeletonize((X > 128)).astype(int), X)
    size = [space1[i+1][0]-space1[i][0] for i in range(len(space1)-1)]
    partitions = [space1[i][0] for i in range(len(space1))]
    if not size:
        output += characterRecognizer(X1)
    else:
        width = min(size)
        for i in range(X1.shape[1]):
            if np.count_nonzero(X1[:, i]) == 1:
                if isValid(i, width, X1.shape[1]) and characterSegmentor(X1[:, (i-width):i], X1[:, i:(i+width)]):
                    partitions.append(i)
        partitions.sort()
        for k in range(len(partitions)-1):
            output += characterRecognizer(X1[:, partitions[k]:partitions[k+1]])
    output += ' '
print(output)
