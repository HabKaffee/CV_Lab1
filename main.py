import os

from src.preprocessing import preprocess_image
from src.lines_extraction import extract_lines_and_intersection_point
from src.utils import get_image, save_image, draw_result


IMAGE_PATH = 'data/road1.png'
OUTPUT_PATH = 'output.png'


if __name__ == '__main__':
    original_image = get_image(IMAGE_PATH)
    image_to_process = preprocess_image(IMAGE_PATH)
    result_lines, intersection_point = extract_lines_and_intersection_point(image_to_process)
    draw_result(original_image, result_lines, intersection_point)
    save_image(OUTPUT_PATH, original_image)
