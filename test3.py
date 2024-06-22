# Importing the required libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from tensorflow.keras.losses import binary_crossentropy
from sklearn.utils.class_weight import compute_class_weight
import shutil

#---------------------------------------Setting up the evaluation parameters---------------------------------------#
# Here, you define functions for the binary cross-entropy loss and the Dice coefficient, which will be used as evaluation metrics during model training.

def binary_cross_entropy_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)

# Function to calculate the dice coefficient
def dice_coef(y_true, y_pred):
    # Flattening the true and predicted masks
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # Calculating the sum of the product of the true and predicted masks and multiplying it with 2.0
    intersection = K.sum(y_true_f * y_pred_f) * 2.0
    # Returning the intersection divided by the sum of the squares of true and predicted masks along with a smoothing factor
    return (intersection + 1e-5) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-5)




# Defining the loss function
def dice_coef_loss(y_true, y_pred):
    # Returning the negative dice coefficient
    return -dice_coef(y_true, y_pred)

#---------------------------------------Importing the dataset---------------------------------------#
# The dataset has already been correctly matched

# Setting the path to the test images with a relative path considering that it is now in : /home/emanuele/Desktop/Remy Robotics/Remy Project/dataset/dataset/test
test_path = r'/home/emanuele/Desktop/Remy Robotics/Remy Project/dataset/dataset/test/images'
# Setting the path to the training images
train_path = os.path.join('/home', 'emanuele', 'Desktop', 'Remy Robotics', 'Remy Project', 'dataset', 'dataset', 'train','images')
# Setting the path to the training masks
train_mask_path = os.path.join('/home', 'emanuele', 'Desktop', 'Remy Robotics', 'Remy Project', 'dataset', 'dataset', 'train','masks')
# Setting the path to the delete folder
delete_path = os.path.join('/home', 'emanuele', 'Desktop', 'Remy Robotics', 'Remy Project', 'dataset', 'dataset','train' ,'delete')


# Function to load the images
def load_train_images(path):
    # Initializing the list to store the images
    images = []
    # Looping over the images until 368 
    for image in tqdm(os.listdir(path)[:368]): #368
        # Constructing the full path to the image
        image_path = os.path.join(path, image)
        
        # Reading the image as 256x256x3
        img = cv2.imread(image_path)
        if img is not None:
            # Resizing the image
            img = cv2.resize(img, (256, 256))
            # Normalizing the image
            img = img / 255.0
            # Appending the image to the list
            images.append(img)
        else:
            print(f"Error reading image: {image_path}")

    # Returning the images
    return images

def load_test_images(path):
    # Initializing the list to store the images
    images = []
    # Looping over the images until 368 
    for image in tqdm(os.listdir(path)[:50]):
        # Constructing the full path to the image
        image_path = os.path.join(path, image)
        
        # Reading the image as 256x256x1
        img = cv2.imread(image_path)
        # Checking if the image was read successfully
        if img is not None:
            # Resizing the image
            img = cv2.resize(img, (256, 256))
            # Normalizing the image
            img = img / 255.0
            # Appending the image to the list
            images.append(img)
        else:
            print(f"Error reading image: {image_path}")

    # Returning the images
    return images

#function to load mask as grayscale images, so they are 256x256x1
def load_masks(path):
    # Initializing the list to store the images
    masks = []
    # Looping over the images until 368 
    for mask in tqdm(os.listdir(path)[:368]): #368
        # Constructing the full path to the image
        mask_path = os.path.join(path, mask)
        
        # Reading the image as 256x256x3
        msk = cv2.imread(mask_path)
        #converting to grayscale: 256x256x1
        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        # Checking if the image was read successfully
        if msk is not None:
            # Resizing the image
            msk = cv2.resize(msk, (256, 256))
            # Normalizing the image
            msk = msk / 255.0
            # Appending the image to the list
            masks.append(msk)
        else:
            print(f"Error reading image: {mask_path}")

    # Returning the images
    return masks


# Loading the test images
print('Loading test images...')
test_images = load_test_images(test_path)
# Loading the training images
print('Loading training images...')
train_images = load_train_images(train_path)
# Loading the training masks
print('Loading training masks...')
train_masks = load_masks(train_mask_path)

# Function to plot the images
def plot_images(images, masks, num_images=10):
    fig, ax = plt.subplots(num_images, 2, figsize=(10, 20))
    for i in range(num_images):
        ax[i][0].imshow(images[i])
        ax[i][1].imshow(masks[i][:, :, 0] ,cmap='gray')

    plt.show()

# Function to convert the images to numpy arrays
def convert_to_numpy(images):
    numpy_images = []
    for image in images:
        image = np.array(image, dtype=np.float32)
        numpy_images.append(image)
    return np.array(numpy_images)

# Function to convert the masks to numpy arrays
def convert_masks_to_numpy(masks):
    numpy_masks = []
    for mask in masks:
        mask = np.array(mask, dtype=np.float32)
        mask = np.expand_dims(mask, axis=-1)
        numpy_masks.append(mask)
    return np.array(numpy_masks)

# Converting the images to numpy arrays
train_images = convert_to_numpy(train_images)
train_masks = convert_masks_to_numpy(train_masks)
test_images = convert_to_numpy(test_images)

plot_images(train_images, train_masks)

# Printing the shape of the training images
print('shapes are:')
print(train_images.shape)
print(train_masks.shape)
print(test_images.shape)

# Splitting the training data into training and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_masks, test_size=0.3)#,random_state=42)

# Printing the shape of the training and validation sets to ensure that everything is correct
print('final shapes are:')
print(train_images.shape)
print(val_images.shape)
print("Train Masks Shape:", train_masks.shape)
print("Validation Masks Shape:", val_masks.shape)

print("Train Masks Data Type:", train_masks.dtype)
print("Validation Masks Data Type:", val_masks.dtype)

print("Train Masks Min Value:", np.min(train_masks))
print("Train Masks Max Value:", np.max(train_masks))

print("Validation Masks Min Value:", np.min(val_masks))
print("Validation Masks Max Value:", np.max(val_masks))

#plot sample val images and val masks
print('plotting val images and val masks')
plot_images(val_images, val_masks)

#plot sample train images and train masks
print('plotting train images and train masks')
plot_images(train_images, train_masks)

#---------------------------------------Setting up the model---------------------------------------#

# Function to create the U-Net model
def create_model():
    inputs = Input((256, 256, 3)) 
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    up6 = Conv2D(512, 2 , activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2 , activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    up8 = Conv2D(128, 2 , activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    up9 = Conv2D(64, 2 , activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)

    # Compiling the model using binary cross-entropy loss and the Adam optimizer with default parameters
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=binary_cross_entropy_loss, metrics=[dice_coef])
    return model
# Creating the model
model = create_model()

#---------------------------------------Setting up training parameters---------------------------------------#

# Plotting the model
plot_model(model, show_shapes=True)

# Printing the summary of the model
#model.summary()

# Initializing the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True) #patience=10

# Initializing the model checkpoint callback
model_checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True)

# Initializing the reduce learning rate on plateau callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, verbose=1, min_lr=1e-6)

# Initializing the tensorboard callback
tensorboard = TensorBoard(log_dir='logs')

# Initializing the image data generator
image_data_generator = ImageDataGenerator(rotation_range=0.5,
                                          width_shift_range=0.5,
                                          height_shift_range=0.5,
                                          shear_range=0.5,
                                          zoom_range=0.5,
                                          #add light augmentation
                                          brightness_range=[-0.5,10.5],
                                          horizontal_flip=True,
                                          vertical_flip=True,
                                          fill_mode='nearest')


#print shapes of train images and train masks before fitting
print(train_images.shape)
print(train_masks.shape)

# Setting the batch size
batch_size = 4  

#---------------------------------------Training the model---------------------------------------#

# Fitting the model on the training data and validating using the validation set
history = model.fit(train_images, train_masks, batch_size=batch_size, epochs=35, validation_data=(val_images, val_masks), callbacks=[early_stopping, model_checkpoint, reduce_lr, tensorboard])#, class_weight=class_weights_dict)

#---------------------------------------Evaluating the model---------------------------------------#

# Evaluating the model on the validation set
results = model.evaluate(val_images, val_masks, batch_size)

#---------------------------------------Plotting the results---------------------------------------#

# Printing the results
print("Loss:", results[0])
print("Validation Dice Coefficient:", results[1])

# Function to plot the predicted masks and the ground truth masks
def plot_images3(images, pred_masks, num_images=10):
    fig, ax = plt.subplots(num_images, 3, figsize=(10, 20))
    for i in range(num_images):
        ax[i][0].imshow(images[i])
        ax[i][1].imshow(pred_masks[i][:, :, 0] ,cmap='gray')

    plt.show()  


# Predicting the masks on the test set
print('Predicting on the test set:',test_images.shape)
pred_masks = model.predict(test_images, verbose=1)
print('Predicting done')

#call plot_images3
plot_images3(test_images, pred_masks)

# Ensure predicted masks are in the expected range and are of the correct data type
print('Fitting done')
print(test_images.shape)
print(pred_masks.shape)
print(pred_masks.dtype)
print(np.min(pred_masks))
print(np.max(pred_masks))


# Function to plot the predicted masks both thresholded and unthresholded
def plot_predicted_masks(images, pred_masks, num_images=10):
    # Initializing the figure
    fig, ax = plt.subplots(num_images, 3, figsize=(15, 30))
    # Looping over the images
    for i in range(num_images):
        # Plotting the image
        ax[i][0].imshow(images[i])
        # Plotting the predicted mask
        ax[i][1].imshow(pred_masks[i])
        #plotting the thresholded predicted mask------------->No need to threshold the predicted mask
        threshold = 0.5
        pred_masks[pred_masks >= threshold] = 1
        pred_masks[pred_masks < threshold] = 0
        ax[i][2].imshow(pred_masks[i])
    # Displaying the plot
    plt.show()

# Plotting the predicted masks using the test set
plot_predicted_masks(test_images, pred_masks)

def overlay_images_masks(images,pred_masks, num_images=10):
    for i in range(num_images):
        plt.figure(figsize=(12, 6))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(images[i])
        plt.title('Original Image')

        # Predicted mask
        plt.subplot(1, 2, 2)
        plt.imshow(pred_masks[i][:, :, 0] ,cmap='gray')
        plt.title('Predicted Mask')

        plt.show()

overlay_images_masks(test_images, pred_masks, num_images=10)


print('shapes are:')
print(val_images.shape)
print(val_masks.shape)

model.save('my_model.keras')

