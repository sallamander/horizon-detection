"""Detect the horizon on the provided images

This script detects the horizon for all images in a user-specified directory,
and saves for each image a plot displaying the detected horizon. Each plot will
be saved in a user-specified directory with the same filename as the original
image.
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt

from utils import detect_horizon_line


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirpath_input_images', type=str, required=True,
        help='Absolute directory path to images to detect the horizon on.'
    )
    parser.add_argument(
        '--dirpath_output_images', type=str, required=True,
        help='Absolute directory path to save output images in.'
    )

    args = parser.parse_args()
    return args


def main():
    """Main logic"""

    args = parse_args()

    dirpath_input_images = args.dirpath_input_images
    dirpath_output_images = args.dirpath_output_images
    msg = ('`dirpath_input_images` and `dirpath_output_images` cannot point to'
           'the same directory.')
    assert dirpath_input_images != dirpath_output_images, msg
    os.mkdir(dirpath_output_images)

    for fname_image in os.listdir(dirpath_input_images):
        fpath_image = os.path.join(dirpath_input_images, fname_image)

        fig, axes = plt.subplots(1, 2)
        image = cv2.imread(fpath_image)
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        horizon_x1, horizon_x2, horizon_y1, horizon_y2 = detect_horizon_line(
            image_grayscale
        )
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title('Original Image')

        axes[1].imshow(image_grayscale, cmap='gray')
        axes[1].axis('off')
        axes[1].plot(
            (horizon_x1, horizon_x2), (horizon_y1, horizon_y2),
            color='r', linewidth='2'
        )
        axes[1].set_title('Grayscaled Image\n with Horizon Line (Red)')

        fpath_save = os.path.join(dirpath_output_images, fname_image)
        fpath_save = fpath_save.replace('jpg', 'png')
        fig.savefig(fpath_save, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()
