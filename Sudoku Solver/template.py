from pipeline import Pipeline

# BEGIN YOUR IMPORTS

# END YOUR IMPORTS
import cv2
from frontalization import gaussian_blur, find_edges, highlight_edges, find_contours, get_max_contour, find_corners, frontalize_image
from recognition import resize_image, get_sudoku_cells
from const import SUDOKU_SIZE
# BEGIN YOUR CODE

"""
create dict of cell coordinates like in this example

CELL_COORDINATES = {"image_0.jpg": {1: (0, 0),
                                    2: (1, 1)},
                    "image_2.jpg": {1: (2, 3),
                                    3: [(2, 1), (0, 4)],
                                    9: (5, 6)}}
"""

CELL_COORDINATES = {
    "image_0.jpg": {
        1: (6, 4),
        2: (2,8),
        3: (3,0),
        4: (6,0),
        5: (4,0),
        6: (4,4),
        7: (4,1),
        8: (0,5),
        9: (2,0)
    },
    "image_5.jpg": {
        1: (3,2),
        2: (2,3),
        3: (1,7),
        4: (0,0),
        5: (3,0),
        6: (5,0),
        7: (0,5),
        8: (3,4),
        9: (3,3)
    }
}

# END YOUR CODE


# BEGIN YOUR FUNCTIONS

# END YOUR FUNCTIONS


def get_template_pipeline():
    # BEGIN YOUR CODE

    pipeline = Pipeline(functions=[gaussian_blur, find_edges, highlight_edges, find_contours, get_max_contour, find_corners, frontalize_image,
                               resize_image, get_sudoku_cells],
                    parameters={"gaussian_blur": {"sigma": 0.42}, # play with the "sigma" parameter
                                "find_corners": {"epsilon": 0.42}, # play with the "epsilon" parameter
                                "resize_image": {"size": SUDOKU_SIZE},
                                # play with the "crop_factor" parameter and binarization_kwargs
                                "get_sudoku_cells": {"crop_factor": [0.9, 0.8], "binarization_kwargs": {'thresh1': 21, 'max_value': 255, 'method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 'thresh2': 25, 'submethod': cv2.THRESH_BINARY}}
                               })
    
    return pipeline

    # END YOUR CODE

    # raise NotImplementedError
