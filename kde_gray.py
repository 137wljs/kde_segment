import cv2
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import os
from tqdm import tqdm

url = "https://img95.699pic.com/xsj/0f/sg/c8.jpg%21/fh/300"                     # Replace with your own image URL
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert('L')
high_res_image = np.array(image)
height, width = high_res_image.shape
print(f"height: {height}, width: {width}")

output_path = os.path.join(os.getcwd(), 'forest_gray.png')
success = cv2.imwrite(output_path, high_res_image)

if success:
    print(f"Image successfully saved at {output_path}")
else:
    print("Failed to save the image")

def kde_foreground_background_segmentation_2d(image):
    pixels_array = np.array(image)
    pixels = []
    pixels = pixels_array.reshape((-1, 1))
    
    kde = KernelDensity(kernel='gaussian', bandwidth=5).fit(pixels)

    chunk_size = 100
    num_chunks = (len(pixels) + chunk_size - 1) // chunk_size
    log_density = np.empty(len(pixels))
    for i in tqdm(range(num_chunks), desc="Calculating KDE", unit="chunk"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(pixels))
        log_density[start_idx:end_idx] = kde.score_samples(pixels[start_idx:end_idx])
    density = np.exp(log_density)
    sorted_density = np.sort(density)
    np.savetxt('density.txt', density)
    percentage = sorted_density[int(0.1 * len(density))]
    print(f"percentage: {percentage}")
    pdf_values = density.reshape((height, width))
    segmented_image = np.where(pdf_values <= percentage, 255, 0)
    
    return segmented_image

segmented_image = kde_foreground_background_segmentation_2d(high_res_image)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(high_res_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

plt.show()
