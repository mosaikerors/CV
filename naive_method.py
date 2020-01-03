import numpy as np
import cv2
import os


def my_bottle(filename):
    """detect bottle
    Args:
        filename: the input filename
    Returns:
        filename: the output filename
        up: the up bottle
        down: the down bottle
        edge_on: the bottle standing cross

        For example:

        './naive-out/img',
        [[[100, 100], 20], [[200, 250], 30],
        [[[150, 150], 20], [[250, 200], 30],
        [[300, 300], [300, 400]] = my_bottle('./images/img')
    """
    img = cv2.imread(filename)
    print(img)
    shape = img.shape
    # for xyx: you will use these two variables
    up = []
    down = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    blur = cv2.medianBlur(binary, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, min(shape[0], shape[1]) // 10, param1=80,
                               param2=20, minRadius=max(shape[0], shape[1]) // 20,
                               maxRadius=min(shape[0], shape[1]) // 10)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # for xyx, please add up and down test
            # add the result to the array: up and down in the format below
            up.append([[i[0], i[1]], i[2]])

    edge_on, result = find_edge_on(img, up, down)

    # draw circles in the result
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 2)

    filename = './naive-out/%s' % filename.split('/')[-1]
    cv2.imwrite(filename, result)
    return filename, up, down, edge_on


def in_circle(circles, point):
    """detect bottle
    Args:
        circles: all the circle in the image
        point: a point in the image
    Returns:
        True or False
    """
    for circle in circles:
        if circle[0][0] - 2 * circle[1] < point[0] < circle[0][0] + 2 * circle[1] \
                and circle[0][1] - 2 * circle[1] < point[1] < circle[0][1] + 2 * circle[1]:
            return True
    return False


# check for close points adding into centers
def add_in_center(centers, point):
    for center in centers:
        if(center[0]-point[0])**2 + (center[1]-point[1])**2 <= 100:
            return centers
    centers.append(point)
    return centers


def find_edge_on(img, up, down):
    centers = []
    blur = cv2.medianBlur(img, 5)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGRA2GRAY)
    _, binary = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
    blur = cv2.medianBlur(binary, 5)
    shape = img.shape
    height, width = shape[0], shape[1]
    window = min(height, width) // 10

    i = 0
    while i < height // window:
        j = 0
        while j < width // window:
            center = [(j * window + min((j + 1) * window, width)) // 2,
                      (i * window + min((i + 1) * window, height)) // 2]
            block = blur[i * window:min((i + 1) * window, height), j * window:min((j + 1) * window, width)]
            if np.sum(block == 255) > window ** 2 // 4 and (not in_circle(up, center)) and (
                    not in_circle(down, center)):
                left_bound, right_bound, up_bound, down_bound = 0, 0, 0, 0
                start = center[0]
                while start >= 0:
                    line = blur[max(0, center[1] - window):min(center[1] + window, height), start]
                    if np.sum(line == 255) < 10:
                        left_bound = start
                        break
                    start -= 1

                start = center[0]
                while start < width:
                    line = blur[max(0, center[1] - window):min(center[1] + window, height), start]
                    if np.sum(line == 255) < 10:
                        right_bound = start
                        break
                    start += 1

                start = center[1]
                while start >= 0:
                    line = blur[start, max(0, center[0] - window):min(center[0] + window, height)]
                    if np.sum(line == 255) < 10:
                        up_bound = start
                        break
                    start -= 1

                start = center[1]
                while start < height:
                    line = blur[start, max(0, center[0] - window):min(center[0] + window, height)]
                    if np.sum(line == 255) < 10:
                        down_bound = start
                        break
                    start += 1

                centers = add_in_center(centers, [(left_bound + right_bound) // 2, (up_bound + down_bound) // 2])
                cv2.line(img, (left_bound, up_bound), (left_bound, down_bound), (255, 0, 0), 3)
                cv2.line(img, (left_bound, up_bound), (right_bound, up_bound), (255, 0, 0), 3)
                cv2.line(img, (right_bound, up_bound), (right_bound, down_bound), (255, 0, 0), 3)
                cv2.line(img, (left_bound, down_bound), (right_bound, down_bound), (255, 0, 0), 3)
            j += 1
        i += 1
    return centers, img


if __name__ == '__main__':
    images = os.listdir('./images')
    for img in images:
        outfile, up, down, edge_on = my_bottle('./images/%s' % img)
        print(outfile, up, down, edge_on)
