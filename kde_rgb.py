import cv2
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import os

url = "https://img95.699pic.com/xsj/0f/sg/c8.jpg%21/fh/300"              # Replace with your own image URL
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert('RGB')
high_res_image = np.array(image)
height, width, _ = high_res_image.shape
print(f"height: {height}, width: {width}")

output_path = os.path.join(os.getcwd(), 'forest_rgb.png')
success = cv2.imwrite(output_path, cv2.cvtColor(high_res_image, cv2.COLOR_RGB2BGR))

if success:
    print(f"Image successfully saved at {output_path}")
else:
    print("Failed to save the image")

def kde_foreground_background_segmentation_2d(image):
    
    pixels_array = np.array(image)
    pixels = pixels_array.reshape((-1, 3))
    
    kde = KernelDensity(kernel='gaussian', bandwidth=5).fit(pixels)
    chunk_size = 100
    num_chunks = (len(pixels) + chunk_size - 1) // chunk_size
    log_density = np.empty(len(pixels))
    for i in tqdm(range(num_chunks), desc=f"Calculating KDE", unit="chunk"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(pixels))
        log_density[start_idx:end_idx] = kde.score_samples(pixels[start_idx:end_idx])
    density = np.exp(log_density)
    np.savetxt('density_rgb.txt', density)
    
    sorted_density = np.sort(density)
    percentage = sorted_density[int(0.1 * len(density))]
    
    pdf_values = density.reshape((height, width))
    segmented_image = np.where(pdf_values <= percentage, 255, 0)
    
    return segmented_image

segmented_image = kde_foreground_background_segmentation_2d(high_res_image)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(high_res_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

plt.show()
