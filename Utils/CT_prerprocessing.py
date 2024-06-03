import os
import cv2
import numpy as np


def preprocess_ct(ct_image_path, save_path):

    # Load the CT image
    ct_image = cv2.imread(ct_image_path, cv2.IMREAD_GRAYSCALE)

    # Upsample the image using bicubic interpolation
    ct_image_upsampled = cv2.resize(ct_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Remove the skull
    # Assuming the skull has higher intensity values compared to the rest of the body
    skull_mask = (ct_image_upsampled > 200).astype(np.uint8)
    ct_image_no_skull = ct_image_upsampled * (1 - skull_mask)

    # Save the preprocessed CT image
    cv2.imwrite(save_path, ct_image_no_skull)


# Example usage
ct_image_path = 'path/to/your/ct_image.png'
save_path = 'path/to/save/preprocessed_ct_image.png'
preprocess_ct(ct_image_path, save_path)