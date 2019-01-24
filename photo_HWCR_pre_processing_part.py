import scipy.io as sc
import numpy as np
import matplotlib.pyplot as plt
import cv2


# Crop the images by 14 px and then resize them back to standard dimensions.
# Finally store the resized image into a commmon array
def crop(data, dim):
    # Array to store the cropped images
    cropped_non_chars_images_array = []

    for i in range(data.shape[0]):
        # Just to keep track of the iteration
        # print(i)

        # Reshape the flatten image data in to the standard dimension
        # and then store the tranpose because after unflattening them
        # the images had horizontal orientation.
        img = data[i].reshape(dim).T

        # The 14 column pixels to crop is generated randomly
        pixels_to_crop = np.random.randint(0, 27, size=14)

        # Only those 14 pixels are selected to crop the image
        img = img[:, pixels_to_crop]

        # And then image is resized into the standard dimensions
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

        # Finally, each individual cropped image is appended to the array
        cropped_non_chars_images_array.append(img)

    return np.asarray(cropped_non_chars_images_array).reshape((-1, 784))


def main():
    # Load the images
    data = sc.loadmat("matlab/emnist-byclass.mat")['dataset']

    # Standard dimension of the images
    dimensions = (28, 28)

    # Get the training data
    train_data = data[0][0][0][0][0][0]

    # Get the testing data
    test_data = data[0][0][1][0][0][0]

    print("Will start cropping the train data")
    # Pre processing on training images
    # and save the non-char images in the below array
    train_non_chars_imgs_arr = crop(train_data, dimensions)

    # Combine both the actual char images as well as generated non-char images so as to create a mixed dataset for Neural Network to classify
    train_data_combined = np.vstack((train_non_chars_imgs_arr, train_data))

    # Generate their labels as zero (0) for non-char images and ones(1) for char images
    train_imgs_labels = np.vstack((np.zeros((train_data.shape[0], 1)), np.ones((train_data.shape[0], 1))))

    print("Now Will start cropping the test data")
    # Pre processing on testing images
    # and save the non-char images in the below array
    test_non_chars_imgs_arr = crop(test_data, dimensions)

    # Combine both the actual char images as well as generated non-char images so as to create a mixed dataset for Neural Network to classify
    test_data_combined = np.vstack((test_non_chars_imgs_arr, test_data))

    # Generate their labels as zero (0) for non-char images and ones(1) for char images
    test_imgs_labels = np.vstack((np.zeros((test_data.shape[0], 1)), np.ones((test_data.shape[0], 1))))
    
    print("Last step....Saving the files")
    # Save both the arrays into a compressed numpy file
    np.savez_compressed("./cropped_non_chars_imgs",
                        shape_of_data=dimensions,
                        train=train_data_combined,
                        test=test_data_combined,
                        train_label=train_imgs_labels,
                        test_label=test_imgs_labels)


if __name__ == '__main__':
    main()