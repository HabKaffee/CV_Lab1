import cv2
import numpy as np
from sklearn.cluster import KMeans


def get_huff_lines(img):
    return cv2.HoughLinesP(img, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)


def clusterize_lines(lines):
    kmeans = KMeans(n_clusters=2) 
    features = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.arctan2(y2 - y1, x2 - x1)
        features.append([x1, y1, x2, y2, length, angle])
    
    kmeans.fit(features)
    clusters = [[] for _ in range(kmeans.n_clusters)]
    [clusters[label].append(line) for label, line in zip(kmeans.labels_, lines)]
    return clusters


def get_middle_line_coordinates(lines):
    starts = np.mean([line[0][:2] for line in lines], axis=0)
    ends = np.mean([line[0][2:] for line in lines], axis=0)
    middle_point = ((starts + ends) / 2).astype(int)

    angle = np.arctan2(ends[1] - starts[1], ends[0] - starts[0])
    length = 320

    delta_x = int(length * np.cos(angle))
    delta_y = int(length * np.sin(angle))

    x1, y1 = middle_point[0] - delta_x, middle_point[1] - delta_y
    x2, y2 = middle_point[0] + delta_x, middle_point[1] + delta_y

    return [x1, y1, x2, y2]


def get_result_lines(clusters):
    return [get_middle_line_coordinates(cluster) for cluster in clusters]


def find_intersection(center_lines):
    (x1, y1, x2, y2), (x3, y3, x4, y4) = center_lines

    m1 = float('inf')
    m2 = float('inf')

    if x2 - x1 != 0:
        m1 = (y2 - y1) / (x2 - x1)
    if x4 - x3 != 0:
        m2 = (y4 - y3) / (x4 - x3)

    if m1 == m2:
        return None

    x = ((m1 * x1 - y1) - (m2 * x3 - y3)) / (m1 - m2)
    y = m1 * (x - x1) + y1

    return int(x), int(y)


def extract_lines_and_intersection_point(image):
    lines = get_huff_lines(image)
    clusters = clusterize_lines(lines)
    result_lines = get_result_lines(clusters)
    intersection_point = find_intersection(result_lines)
    return result_lines, intersection_point
