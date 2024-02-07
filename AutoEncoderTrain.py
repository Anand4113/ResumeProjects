import numpy as np
import os
from skimage.util import random_noise
from skimage.transform import resize
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Nadam

# Function to add noise to an image
def add_noise_to_image(image, noise_mode='gaussian', var=0.05):
    noisy_image = np.clip(random_noise(image, mode=noise_mode, var=var), 0, 1)
    return noisy_image

# Function to normalize the image to the range [0, 1]
def normalize_image(image):
    return image / 255.0

# Function to load images from a directory
def load_images_from_directory(image_dir, size=(256, 256)):
    original_images = []
    noisy_images = []
    for file in os.listdir(image_dir):
        if file.endswith('.jpg'):
            image_path = os.path.join(image_dir, file)
            image = Image.open(image_path)
            image = image.convert('L')  # Convert to grayscale
            image = np.array(image)
            image_resized = resize(image, size, anti_aliasing=True)
            original_images.append(normalize_image(image_resized))
            noisy_images.append(normalize_image(add_noise_to_image(image_resized)))
    return np.array(original_images), np.array(noisy_images)

# Autoencoder model for denoising
def autoencoder(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Encoder
    x = Conv2D(32, (3, 3), activation='elu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='elu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(1, (3, 3), activation='elu', padding='same')(x)

    model = Model(inputs, decoded)
    model.compile(optimizer=Nadam(), loss='mean_squared_error', metrics=['accuracy'])

    return model

# Directory containing your MRI images
image_dir = r'C:\Users\allen\Desktop\Training\glioma'

# Load the dataset
original_images, noisy_images = load_images_from_directory(image_dir)

# Preprocessing for the model (reshape if necessary)
original_images = np.expand_dims(original_images, axis=-1)
noisy_images = np.expand_dims(noisy_images, axis=-1)

# Initialize the Autoencoder model
autoencoder_model = autoencoder(input_size=(256, 256, 1))

# Train the model with the dataset
autoencoder_model.fit(noisy_images, original_images, batch_size=16, epochs=25)  # Adjust batch size and epochs as necessary

# After training, save the model to disk
autoencoder_model.save('autoencoder_model_1123.h5')