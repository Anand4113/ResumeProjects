import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from skimage.transform import resize
from PIL import Image
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Function to add noise to an image
def add_noise_to_image(image, noise_mode='gaussian', var=0.05):
    noisy_image = random_noise(image, mode=noise_mode, var=var)
    return noisy_image

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
            original_images.append(image_resized)
            noisy_images.append(add_noise_to_image(image_resized))
    return np.array(original_images), np.array(noisy_images)

# U-Net model (unchanged from your provided code)
def unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Contracting Path
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    # Expanding Path
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1], axis=3)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Calculate PSNR, SSIM, and MSE
def calculate_metrics(original, denoised):
    original = original.astype('float32') / 255
    denoised = denoised.astype('float32') / 255
    data_range = 1.0  # The data range of the images; this assumes the images have been scaled to [0, 1]
    p = psnr(original, denoised, data_range=data_range)
    s = ssim(original, denoised, data_range=data_range)
    m = mse(original, denoised)
    return p, s, m

# Directory containing your MRI images
image_dir = r"C:\Users\allen\Desktop\Training\glioma"

# Load the dataset
original_images, noisy_images = load_images_from_directory(image_dir)

# Preprocessing for the model (reshape if necessary)
original_images = np.expand_dims(original_images, axis=-1)
noisy_images = np.expand_dims(noisy_images, axis=-1)

# Initialize the U-Net model
unet_model = unet(input_size=(256, 256, 1))

# Train the model with the dataset
unet_model.fit(noisy_images, original_images, batch_size=16, epochs=50)  # Adjust batch size and epochs as necessary

# After training, save the model to disk
unet_model.save('unet_model.h5')