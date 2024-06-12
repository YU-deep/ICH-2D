import os
import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk


def resample_size(sitk_image, depth):
    """
    Resamples the image to the specified depth.
    
    Parameters:
    sitk_image (sitk.Image): The SimpleITK image.
    depth (int): Target depth.
    
    Returns:
    sitk.Image: The resampled image.
    """
    euler3d = sitk.Euler3DTransform()  # 3D Euler transform for rigid registration
    size = sitk_image.GetSize()
    spacing = sitk_image.GetSpacing()
    
    width, height, _ = size
    xspacing, yspacing, zspacing = spacing
    print(spacing)
    new_spacing_z = zspacing / (depth / float(height))
    print(new_spacing_z)
    origin = sitk_image.GetOrigin()
    direction = sitk_image.GetDirection()
    
    # Calculate new size based on new spacing
    new_size = (width, height, depth)
    new_spacing = (xspacing, yspacing, new_spacing_z)
    
    # Perform resampling
    sitk_image = sitk.Resample(sitk_image, new_size, euler3d, sitk.sitkNearestNeighbor, origin, new_spacing, direction)
    return sitk_image


def preprocess_nii(img_path):
    """
    Reads a NIfTI format image, sets the intensity window, and rescales intensity values to the range 0-255.
    
    Parameters:
    img_path (str): Path to the NIfTI image file.
    
    Returns:
    sitk.Image: The preprocessed image.
    """
    # Read the NIfTI image
    sitkImage = sitk.ReadImage(img_path)
    
    # Set intensity window levels
    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    intensityWindowingFilter.SetWindowMaximum(400)  # Window maximum value, chosen as 400
    intensityWindowingFilter.SetWindowMinimum(15)   # Window minimum value, chosen as 15
    intensityWindowingFilter.SetOutputMaximum(255)  # Map output maximum to 255
    intensityWindowingFilter.SetOutputMinimum(0)    # Map output minimum to 0
    
    # Resample the image size
    sitkImage = resampleSize(sitkImage, 32)
    
    # Apply the intensity windowing
    sitkImage = intensityWindowingFilter.Execute(sitkImage)
    return sitkImage

def remove_skull(image):
    """
    Removes the skull part from the CT image.
    
    Parameters:
    image (sitk.Image): The preprocessed image.
    
    Returns:
    sitk.Image: The skull-stripped image.
    """
    data = sitk.GetArrayFromImage(image)

    # Thresholding to isolate skull
    threshold = 25
    binary_image = data < threshold

    binary_image = ndimage.binary_opening(binary_image)
    label_image, num_labels = ndimage.label(binary_image)

    largest_label = np.argmax(np.bincount(label_image.flat)[1:]) + 1

    brain_mask = label_image == largest_label

    skull_stripped_data = data * brain_mask
                
    skull_stripped_image = sitk.GetImageFromArray(skull_stripped_data)
    skull_stripped_image.CopyInformation(image)
    return skull_stripped_image


def get_mean_and_std(good_path, bad_path):
    """
    Computes the mean and standard deviation of "good" and "bad" image datasets.
    
    Parameters:
    good_path (str): Path to the folder containing "good" images.
    bad_path (str): Path to the folder containing "bad" images.
    
    Returns:
    tuple: Overall mean and standard deviation.
    """
    good = os.listdir(good_path)
    bad = os.listdir(bad_path)
    good_len = len(good)
    bad_len = len(bad)
    
    good_img_list = []
    for item in good:
        img = sitk.ReadImage(os.path.join(good_path, item))
        img_array = sitk.GetArrayFromImage(img).flatten()
        good_img_list = np.concatenate((good_img_list, img_array), axis=0)
        print(item)
    
    good_mean = np.mean(good_img_list)
    good_std = np.std(good_img_list)
    
    bad_img_list = []
    for item in bad:
        img = sitk.ReadImage(os.path.join(bad_path, item))
        img_array = sitk.GetArrayFromImage(img).flatten()
        bad_img_list = np.concatenate((bad_img_list, img_array), axis=0)
        print(item)
    
    bad_mean = np.mean(bad_img_list)
    bad_std = np.std(bad_img_list)
    
    mean = (good_mean * good_len + bad_mean * bad_len) / (good_len + bad_len)
    a = (good_len - 1) * (good_std ** 2) + (bad_len - 1) * (bad_std ** 2) + good_len * bad_len / (good_len + bad_len) * (good_mean ** 2 + bad_mean ** 2 - 2 * good_mean * bad_mean)
    std = (a / (good_len + bad_len - 1)) ** 0.5
    
    return mean, std


def normalize_image(image, mean, std):
    """
    Normalizes the image using the provided mean and standard deviation.
    
    Parameters:
    image (sitk.Image): The image to be normalized.
    mean (float): The mean value for normalization.
    std (float): The standard deviation value for normalization.
    
    Returns:
    sitk.Image: The normalized image.
    """
    img_array = sitk.GetArrayFromImage(image)
    normalized_array = (img_array - mean) / std
    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image)
    return normalized_image


def preprocess(raw_img_in, mean, std):
    """
    Integrates all preprocessing steps: intensity windowing, skull stripping, registration, and normalization.
    
    Parameters:
    raw_img_in (str): Path to the raw input NIfTI image.
    template_img_path (str): Path to the template image for registration.
    mean (float): The mean value for normalization.
    std (float): The standard deviation value for normalization.
    
    Returns:
    sitk.Image: The fully preprocessed image.
    """
    # Preprocess the image (intensity windowing and resampling)
    preprocessed_img = preprocess_nii(raw_img_in)
    
    # Remove skull from the image
    skull_stripped_img = remove_skull(preprocessed_img)
    
    # Normalize the registered image
    normalized_img = normalize_image(skull_stripped_img, mean, std)
    
    return normalized_img


# Example usage
good_path = 'path_to_good_images'
bad_path = 'path_to_bad_images'
mean, std = get_mean_and_std(good_path, bad_path)

img_out = preprocess('path_to_raw_image.nii', mean, std)  # mean = 2.4908, std = 6.2173
