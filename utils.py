from pathlib import Path
from skimage import io
import tifffile
import os
import napari
import pyclesperanto_prototype as cle
from skimage.measure import regionprops_table
import pandas as pd
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import numpy as np


def read_images(directory_path):
    """Reads all the images in the input path and organizes them according to the well_id"""
    # Define the directory containing your files
    directory_path = Path(directory_path)

    # Initialize a dictionary to store the grouped (per position) files
    images_per_position = {}

    # Iterate through the files in the directory
    for file_path in directory_path.glob("*"):
        # Check if the path is a file and ends with ".tif"
        if file_path.is_file() and file_path.suffix.lower() == ".tif":
            # Get the filename without the extension
            filename = file_path.stem
            # Remove unwanted files (Plate_R files)
            if "ch02" in filename:
                pass
            # Remove any other unwanted files
            elif "_z" not in filename:
                pass
            else:
                # Extract the last part of the filename (e.g., 1_Crop001_z00_ch00)
                last_part = filename.split(" ")[1]

                # Get the first three letters to create the group name (position_id)
                position_id = last_part[:1]

                # Check if the well_id exists in the dictionary, if not, create a new list
                if position_id not in images_per_position:
                    images_per_position[position_id] = []

                # Append the file to the corresponding group
                images_per_position[position_id].append(str(file_path))

    return images_per_position


def create_stack(image_paths):
    """Takes a collection of image paths containing individual z-stacks and returns a stack of images"""
    # Load images from the specified paths
    image_collection = io.ImageCollection(image_paths)
    # Stack images into a single 3D numpy array
    stack = io.concatenate_images(image_collection)

    return stack


def return_stacks_per_position_id(images_per_position, position_id):
    """Takes a list of paths defined by position_id and stored in images_per_position, stacks them on a per channel basis and returns the stacks"""

    # Format the position_id as a string
    position_id = str(position_id)

    # Define empty lists to store the list of paths containing each individual z-slice
    ch00_paths = []
    ch01_paths = []

    # Loop over each path stored under the position_id key under images_per_position dictionary
    for image_path in images_per_position[position_id]:
        if "ch00" in image_path:
            ch00_paths.append(image_path)
        elif "ch01" in image_path:
            ch01_paths.append(image_path)

    # Generate the stacks
    ch00_stack = create_stack(ch00_paths)
    ch01_stack = create_stack(ch01_paths)

    return ch00_stack, ch01_stack


def make_isotropic(image, scaling_x_um, scaling_y_um, scaling_z_um):
    """Scale the image with the voxel size used as scaling factor to get an image stack with isotropic voxels"""
    # Set the x and y scaling to 1 to avoid resizing (compressing the input image), this way the rescaling factor is only applied to z
    multiplier = 1 / scaling_x_um

    scaling_x_um = scaling_x_um * multiplier
    scaling_y_um = scaling_y_um * multiplier
    scaling_z_um = scaling_z_um * multiplier

    image_resampled = cle.scale(
        image,
        factor_x=scaling_x_um,
        factor_y=scaling_y_um,
        factor_z=scaling_z_um,
        auto_size=True,
    )

    return image_resampled


def extract_nuclei_labels(
    ch00_stack,
    scaling_x_um=0.342,
    scaling_y_um=0.342,
    scaling_z_um=0.663,
    top_hat_radius=5,
    gaussian_sigma=2,
    voronoi_otsu_spot_sigma=10,
    voronoi_otsu_outline_sigma=1,
    closing_labels_radius=5,
    erosion_labels_radius=3,
):
    """Takes a nuclei stack and analysis parameters as inputs and outputs a .cle OCLarray as output containing the post-processed nuclei labels"""

    # Rescale the ch00_stack (nuclei) and ch01_stack to make data isotropic in order to apply voronoi_otsu labeling afterwards
    nuclei_resampled = make_isotropic(
        ch00_stack, scaling_x_um, scaling_y_um, scaling_z_um
    )

    # Remove background with a top_hat_filter
    background_subtracted = cle.top_hat_box(
        nuclei_resampled,
        radius_x=top_hat_radius,
        radius_y=top_hat_radius,
        radius_z=top_hat_radius,
    )

    # Apply gaussian blur to prevent the formation of holes upon labeling
    post_gaussian = cle.gaussian_blur(
        background_subtracted,
        sigma_x=gaussian_sigma,
        sigma_y=gaussian_sigma,
        sigma_z=gaussian_sigma,
    )

    # Voronoi-Otsu labeling
    segmented = cle.voronoi_otsu_labeling(
        post_gaussian,
        spot_sigma=voronoi_otsu_spot_sigma,
        outline_sigma=voronoi_otsu_outline_sigma,
    )

    # Close holes in labels to avoid false emtpy volumes within the nuclei
    closed_labels = cle.closing_labels(segmented, radius=closing_labels_radius)

    # Erode the labels to avoid quantifying protein of interest spots in close proximity but not inside the nucleus
    eroded_labels = cle.erode_labels(closed_labels, radius=erosion_labels_radius)

    # In order to remove labels on edges I have to remove the labels from the first and last z-planes
    # Otherwise upon applying remove_labels_on_edges (skimage.clear_border) the nuclei touching "up" and "down" will be removed

    # Pull eroded_labels from GPU to manipulate it as a np.array
    eroded_labels_np = cle.pull(eroded_labels)

    # Set the first and last slices to 0
    eroded_labels_np[0, :, :] = 0
    eroded_labels_np[-1, :, :] = 0

    # Once the top and bottom slices are empty (no labels) we can apply remove on edges
    remove_border_labels = nsbatwm.remove_labels_on_edges(eroded_labels_np)

    return remove_border_labels


def measure_intensity(nuclei_labels, marker_resampled):
    """Takes the resampled (isotropic) nuclei labels and marker_resampled (intensity image) as inputs and outputs a dataframe with per nucleus stats"""
    # Transform cle. OCLarrays into np.arrays for further processing using skimage.regionprops
    nuclei_labels_np = cle.pull(nuclei_labels)
    marker_resampled_np = cle.pull(marker_resampled)

    # Extract regionprops from labels not touching the borders
    props = regionprops_table(
        label_image=nuclei_labels_np,
        intensity_image=marker_resampled_np,
        properties=[
            "label",
            "intensity_mean",
            "intensity_max",
            "centroid",
            "area_filled",
        ],
    )

    # Construct a dataframe
    df = pd.DataFrame(props)

    return df


def save_stacks(images_per_position, output_dir="./output/processed_stacks"):
    """Takes a images_per_position from read_images as input, stacks them on a per channel basis and saves the resulting images on a per position basis"""
    for position_id, files in images_per_position.items():

        ch00_paths = []
        ch01_paths = []

        for image_path in images_per_position[position_id]:
            if "ch00" in image_path:
                ch00_paths.append(image_path)
            elif "ch01" in image_path:
                ch01_paths.append(image_path)

        # Generate the stacks
        ch00_stack = create_stack(ch00_paths)
        ch01_stack = create_stack(ch01_paths)

        # Create a directory to store the tif files if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Construct the output file path
        output_path_ch00 = os.path.join(output_dir, f"Position {position_id}_ch00.tif")
        output_path_ch01 = os.path.join(output_dir, f"Position {position_id}_ch01.tif")

        # Save the resulting minimum projection
        tifffile.imwrite(output_path_ch00, ch00_stack)
        tifffile.imwrite(output_path_ch01, ch01_stack)
