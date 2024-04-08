import cv2


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def isolate_road_marking(image):
    _, mask = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(image, image, mask=mask)


def apply_blur(image, kernel_size):
    return cv2.GaussianBlur(image, kernel_size, 0)


def get_image_canny(image):
    return cv2.Canny(image, 240, 250)


def preprocess_image(path_to_image):
    image = cv2.imread(path_to_image)
    image = convert_to_grayscale(image)
    image = isolate_road_marking(image)
    # apply blur 7 times to get rid of background mostly
    for _ in range(7):
        image = apply_blur(image, (5,5))
    image = get_image_canny(image)
    return image
