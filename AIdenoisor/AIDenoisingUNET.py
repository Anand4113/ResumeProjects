import numpy as np
from skimage.util import random_noise
from skimage.transform import resize
from PIL import Image
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import tkinter as tk
from tkinter import filedialog, Label
import matplotlib.pyplot as plt

# Function to add noise to an image at different levels
def add_noise_to_image(image, noise_level='low'):
    if noise_level == 'low':
        var = 0.01
    elif noise_level == 'medium':
        var = 0.05
    elif noise_level == 'high':
        var = 0.1
    else:
        raise ValueError("Invalid noise level. Choose 'low', 'medium', or 'high'.")
    
    noisy_image = random_noise(image, mode='gaussian', var=var)
    return noisy_image

# Function to calculate evaluation metrics
def calculate_metrics(original, denoised):
    if original.max() > 1.0:
        original = original / 255.0
    if denoised.max() > 1.0:
        denoised = denoised / 255.0

    original = original.astype('float32')
    denoised = denoised.astype('float32')

    psnr_value = psnr(original, denoised, data_range=1.0)
    ssim_value = ssim(original, denoised, data_range=1.0)
    mse_value = mse(original, denoised)

    return psnr_value, ssim_value, mse_value

# Function to normalize an image
def normalize_image(image):
    return image / np.max(image)

# Load the pre-trained model
unet_model = load_model(r'C:\Users\allen\unet_model.h5')

# GUI function to denoise images
def gui_denoise_images():
    image_path = filedialog.askopenfilename()
    if not image_path:
        return

    image = Image.open(image_path)
    image = image.convert('L')
    mri_image = np.array(image)
    mri_image_resized = resize(mri_image, (256, 256), anti_aliasing=True)

    metrics = {}
    noise_levels = ['low', 'medium', 'high']

    # Create a figure with 3 rows and 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))

    # Add a main title to the figure
    fig.suptitle("AI Denoisor for Brain MRIs - PAAR Denoisors", fontsize=16)

    # Display the original image
    axes[0, 1].imshow(mri_image_resized, cmap='gray')
    axes[0, 1].set_title('Original Image')
    axes[0, 1].axis('off')

    # Process and display noisy and denoised images
    for i, level in enumerate(noise_levels):
        noisy_img = add_noise_to_image(mri_image_resized, noise_level=level)
        normalized_noisy_img = normalize_image(noisy_img)
        noisy_img_reshaped = np.expand_dims(normalized_noisy_img, axis=0)
        noisy_img_reshaped = np.expand_dims(noisy_img_reshaped, axis=-1)
        denoised_img = unet_model.predict(noisy_img_reshaped)[0, :, :, 0]
        
        # Calculate metrics for the denoised image
        psnr_value, ssim_value, mse_value = calculate_metrics(mri_image_resized, denoised_img)
        metrics[level] = (psnr_value, ssim_value, mse_value)
        
        # Display the noisy image
        axes[1, i].imshow(noisy_img, cmap='gray')
        axes[1, i].set_title(f'Noisy ({level})')
        axes[1, i].axis('off')
        
        # Display the denoised image with metrics
        axes[2, i].imshow(denoised_img, cmap='gray')
        axes[2, i].set_title(f'Denoised ({level})')
        axes[2, i].text(0.5, -0.55, f'PSNR: {psnr_value:.2f}\nSSIM: {ssim_value:.4f}\nMSE: {mse_value:.4f}', 
                         size=12, ha="center", transform=axes[2, i].transAxes)

    # Hide the empty subplots
    for ax in axes[0, [0, 2]]:
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    denoised_img = unet_model.predict(noisy_img_reshaped)[0, :, :, 0]

# GUI setup
root = tk.Tk()
root.title("MRI Denoiser")

# Add the main title
main_title = Label(root, text="AI Denoisor for Brain MRIs", font=("Helvetica", 16))
main_title.pack()

# Add the second title
second_title = Label(root, text="PAAR Denoisors", font=("Helvetica", 14))
second_title.pack()

# Button for denoising images
denoise_button = tk.Button(root, text="Denoise Images", command=gui_denoise_images)
denoise_button.pack()

root.mainloop()