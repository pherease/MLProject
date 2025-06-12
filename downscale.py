import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os



def getImage(path):
    Image = plt.imread(path)
    return Image

def downscale(image, outputDims):
    """
    Downscales an image to the specified dimensions using nearest neighbor interpolation.
    
    Parameters:
    - image: The input image as a NumPy array.
    - outputDims: A tuple (new_height, new_width) specifying the desired dimensions.
    
    Returns:
    - downscaled_image: The downscaled image as a NumPy array.
    """
    height, width = image.shape[:2]
    new_height, new_width = outputDims
    
    # Calculate the scaling factors
    row_scale = height / new_height
    col_scale = width / new_width
    
    # Create an empty array for the downscaled image
    downscaled_image = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    
    for i in range(new_height):
        for j in range(new_width):
            # Find the corresponding pixel in the original image
            orig_i = int(i * row_scale)
            orig_j = int(j * col_scale)
            downscaled_image[i, j] = image[orig_i, orig_j]
    
    return downscaled_image


def main():
    outputDims = (64, 64)  # Desired output dimensions
    df = pd.read_csv('data.csv')
    image_paths = df['filePath']
    for path in image_paths:
        image = getImage(path)
        downscaled_image = downscale(image, outputDims)
        savePath = path.replace('data', 'data64x64')
        savePath = savePath.replace('.jpg', '_64x64.jpg')
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        plt.imsave(savePath, downscaled_image)
    print("Downscaling completed and images saved.")

if __name__ == "__main__":
    main()