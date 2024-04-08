import cv2


def draw_lines(img, lines):
    if not lines:
        return
    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 10)


def draw_intersection_point(img, intersection_point):
    cv2.circle(img, intersection_point, 10, (0, 128, 255), -1)


def draw_result(img, lines, intersection_point):
    draw_lines(img, lines)
    draw_intersection_point(img, intersection_point)


def get_image(path_to_file):
    return cv2.imread(path_to_file)


def save_image(path_to_file, img):
    return cv2.imwrite(path_to_file, img)
