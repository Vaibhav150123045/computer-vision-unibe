import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from utils import read_image, show_image
from const import NUM_CELLS, CELL_SIZE, SUDOKU_SIZE
from utils import load_templates


# BEGIN YOUR IMPORTS

# END YOUR IMPORTS


# BEGIN YOUR FUNCTIONS

# END YOUR FUNCTIONS


def resize_image(image, size):
    """
    Args:
        image (np.array): input image of shape [H, W]
        size (int, int): desired image size
    Returns:
        resized_image (np.array): 8-bit (with range [0, 255]) resized image
    """
    # BEGIN YOUR CODE

    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    return resized_image
    
    # END YOUR CODE

    # raise NotImplementedError


def binarize(image, **binarization_kwargs):
    """
    Args:
        image (np.array): input image
        binarization_kwargs (dict): dict of parameter values
    Returns:
        binarized_image (np.array): binarized image

    You can find information about different thresholding algorithms here
    https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    """
    # BEGIN YOUR CODE

    
    max_value = binarization_kwargs.get('max_value', 255)
    method = binarization_kwargs.get('method', cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    submethod = binarization_kwargs.get('submethod', cv2.THRESH_BINARY)
    thresh1 = binarization_kwargs.get('thresh1', 15)
    thresh2 = binarization_kwargs.get('thresh2', 25)
    # _, binarized_image = cv2.threshold(image, thresh, max_value, method)
    binarized_image = cv2.adaptiveThreshold(image, max_value, method, submethod, thresh1, thresh2)
    
    return binarized_image
    
    # END YOUR CODE

    # raise NotImplementedError


def crop_image(image, crop_factor):
    size = image.shape[:2]
    
    cropped_size = (int(size[0]*crop_factor[0]), int(size[1]*crop_factor[1]))
    shift = ((size[0] - cropped_size[0]) // 2, (size[1] - cropped_size[1]) // 2)

    cropped_image = image[shift[0]:shift[0]+cropped_size[0],
                          shift[1]:shift[1]+cropped_size[1]]

    return cropped_image


def get_sudoku_cells(frontalized_image, crop_factor=0.42, binarization_kwargs={}):
    """
    Args:
        frontalized_image (np.array): frontalized sudoku image
        crop_factor (float): how much cell area we should preserve
        binarization_kwargs (dict): dict of parameter values for the binarization function
    Returns:
        sudoku_cells (np.array): array of num_cells x num_cells sudoku cells of shape [N, N, S, S]
    """
    # BEGIN YOUR CODE

    resized_image = resize_image(frontalized_image, SUDOKU_SIZE)
    
    binarized_image = binarize(image = resized_image, thresh1 = binarization_kwargs['thresh1'], max_value = binarization_kwargs['max_value'], submethod = binarization_kwargs['submethod'], method = binarization_kwargs['method'], thresh2 = binarization_kwargs['thresh2'])
    
    sudoku_cells = np.zeros((NUM_CELLS, NUM_CELLS, *CELL_SIZE), dtype=np.uint8) 
    for i in range(NUM_CELLS):
        for j in range(NUM_CELLS):
            sudoku_cell = binarized_image[64*i:64*(i+1), 64*j:64*(j+1)]
            sudoku_cell = crop_image(sudoku_cell, crop_factor=crop_factor)
            sudoku_cells[i, j] = resize_image(sudoku_cell, CELL_SIZE)

    return sudoku_cells

    # END YOUR CODE

    # raise NotImplementedError


def is_empty(sudoku_cell, **kwargs):
    """
    Args:
        sudoku_cell (np.array): image (np.array) of a Sudoku cell
        kwargs (dict): dict of parameter values for this function
    Returns:
        cell_is_empty (bool): True or False depends on whether the Sudoku cell is empty or not
    """
    # BEGIN YOUR CODE

    sudoku_cell = crop_image(sudoku_cell, crop_factor=[0.5,0.5])
    # cell_is_empty = np.all(sudoku_cell == 255)

    total_pixels = sudoku_cell.size

    # Step 4: Count the number of white pixels (pixel value = 255)
    white_pixels = np.sum(sudoku_cell == 255)

    # Step 5: Calculate the percentage of white pixels
    white_pixel_percentage = white_pixels / total_pixels

    # Step 6: Check if the percentage of white pixels is greater than the threshold (90%)
    cell_is_empty = white_pixel_percentage >= 0.96

    
    return cell_is_empty

    # END YOUR CODE

    # raise NotImplementedError


def get_digit_correlations(sudoku_cell, templates_dict):
    """
    Args:
        sudoku_cell (np.array): image (np.array) of a Sudoku cell
        templates_dict (dict): dict with digits as keys and lists of template images (np.array) as values
    Returns:
        correlations (np.array): an array of correlation coefficients between Sudoku cell and digit templates
    """
    correlations = np.zeros(9)

    # BEGIN YOUR CODE
    
    if is_empty(sudoku_cell):
        return correlations

    for digit, templates in templates_dict.items():
        # calculate the correlation score between the sudoku_cell and a digit
        max_corr = 0
        for template in templates:
            template = crop_image(template, [0.78, 0.625])
            result = cv2.matchTemplate(sudoku_cell, template, cv2.TM_CCORR_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(crop_image(result, crop_factor=[1, 1]))
            if( max_val > max_corr):
                max_corr = max_val

        
        correlations[digit - 1] = max_corr
    
    return correlations
    
    # END YOUR CODE

    # raise NotImplementedError


def show_correlations(sudoku_cell, correlations):
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    
    show_image(sudoku_cell, axis=axes[0], as_gray=True)
    
    colors = ['blue' if value < np.max(correlations) else 'red' for value in correlations]
    axes[1].bar(np.arange(1, 10), correlations, tick_label=np.arange(1, 10), color=colors)
    axes[1].set_title("Correlations")


def recognize_digits(sudoku_cells, templates_dict, threshold=0.5):
    """
    Args:
        sudoku_cells (np.array): np.array of the Sudoku cells of shape [N, N, S, S]
        templates_dict (dict): dict with digits as keys and lists of template images (np.array) as values
        threshold (float): empty cell detection threshold
    Returns:
        sudoku_matrix (np.array): a matrix of shape [N, N] with recognized digits of the Sudoku grid
    """
    sudoku_matrix = np.zeros(sudoku_cells.shape[:2], dtype=np.uint8)
    
    # BEGIN YOUR CODE
    
    for i in range(sudoku_cells.shape[0]):
        for j in range(sudoku_cells.shape[1]):
            sudoku_cell = sudoku_cells[i][j]
            is_cell_empty = is_empty(sudoku_cell)
            if is_cell_empty:
                sudoku_matrix[i, j] = 0
            else:
                correlations = get_digit_correlations(sudoku_cell, templates_dict)
                index_with_max_correlation = np.argmax(correlations)
                sudoku_matrix[i, j] = index_with_max_correlation + 1
            

    return sudoku_matrix

    # END YOUR CODE

    raise NotImplementedError


def show_recognized_digits(image_paths, pipeline, figsize=(16, 12), digit_fontsize=10):
    nrows = len(image_paths) // 4 + 1
    ncols = 4
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if len(axes.shape) == 1:
        axes = axes[np.newaxis, ...]

    for j in range(len(image_paths), nrows * ncols):
        axis = axes[j // ncols][j % ncols]
        show_image(np.ones((1, 1, 3)), axis=axis)
    
    for index, image_path in enumerate(tqdm(image_paths)):
        axis = axes[index // ncols][index % ncols]
        axis.set_title(os.path.split(image_path)[1])
        
        sudoku_image = read_image(image_path=image_path)
        frontalized_image, sudoku_cells = pipeline(sudoku_image)

        templates_dict = load_templates()
        sudoku_matrix = recognize_digits(sudoku_cells, templates_dict)

        show_image(frontalized_image, axis=axis, as_gray=True)
        
        frontalized_cell_size = (frontalized_image.shape[0]//NUM_CELLS, frontalized_image.shape[1]//NUM_CELLS)
        for i in range(NUM_CELLS):
            for j in range(NUM_CELLS):
                axis.text((j + 1)*frontalized_cell_size[0] - int(0.3*frontalized_cell_size[0]),
                          i*frontalized_cell_size[1] + int(0.3*frontalized_cell_size[1]),
                          str(sudoku_matrix[i, j]), fontsize=digit_fontsize, c='r')


def show_solved_sudoku(frontalized_image, sudoku_matrix, sudoku_matrix_solved, digit_fontsize=20):
    show_image(frontalized_image, as_gray=True)

    frontalized_cell_size = (frontalized_image.shape[0]//NUM_CELLS, frontalized_image.shape[1]//NUM_CELLS)
    for i in range(NUM_CELLS):
        for j in range(NUM_CELLS):
            if sudoku_matrix[i, j] == 0:
                plt.text(j*frontalized_cell_size[0] + int(0.3*frontalized_cell_size[0]),
                         (i + 1)*frontalized_cell_size[1] - int(0.3*frontalized_cell_size[1]),
                         str(sudoku_matrix_solved[i, j]), fontsize=digit_fontsize, c='g')
