import cv2
import numpy as np
from draw import draw_polygon
import os


def get_GT_vector(img_id):
    '''
    This function returns the GT_vector by detecting whether each strip contains a weed or not The process goes like
    this: Finding the segmentation coordinates Creating an empty mask with the same dimensions as the image Adding
    the segmentation coordinates to the mask and filling it in white (255) #Check the sum of pixels in each strip;
    if the sum is larger than 0 then there's a weed -> GT_vector value in that strip is 1
    '''

    GT_vector = [0] * 6

    # Loading the image
    image = cv2.imread(f'test/images/{img_id}.jpeg')
    print(image)
    # Loading the segmentation coordinates
    with open(f'test/labels/{img_id}.txt', 'r') as f:
        segmentation = [list(map(float, line.strip().split())) for line in f]
    coordinates = []
    for segment in segmentation:
        # Reshape the segment into an array of points
        if not segment:
            continue
        class_id = segment[0]
        # Excluding soy
        if class_id == 3:
            continue
        print(class_id)
        #label = id_2_cls[int(class_id)]
        X = (np.array(segment[1:]).reshape((-1,2))[:,0]*1440).astype(np.int32)
        Y = (np.array(segment[1:]).reshape((-1,2))[:,1]*192).astype(np.int32)
        points = np.column_stack((X, Y))
        coordinates.append(points)
    # print(coordinates)

    # Creating an empty mask with the same dimensions as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    print(mask.shape)
    # Fill the polygon defined by the coordinates with white (255)
    cv2.fillPoly(mask, coordinates, 255)
    #cv2.imshow('mask',mask)
    ## Check the sum of pixels in each strip; if the sum is larger than 0 then there's a weed -> GT_vector is 1
    for i in range(6):
        if np.sum(mask[0:192, i*240:(i+1)*240]) > 0:
            GT_vector[i] = 1

    return GT_vector


def draw_GT_vector_with_seg(file_id, GT_Vector, id_2_cls):
    '''
    This function returns an image divided into strips, with the segmentation and GT_vector drawn on it
    '''

    # Loading the image
    image = cv2.imread(f'test/images/{file_id}.jpeg')

    # Iterating through GT_Vector and drawing text on the image
    for i, text in enumerate(GT_Vector):
        position = (i * 240 + 10, 170)
        cv2.putText(image, str(text), position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    label_path = os.path.join('test', 'labels', f'{file_id}.txt')

    # Load the segmentation label
    with open(label_path, 'r') as f:
        segmentation = [list(map(float, line.strip().split())) for line in f]

    # Drawing the segmentation
    for segment in segmentation:
        # Reshape the segment into an array of points
        if not segment:
            continue
        class_id = segment[0]
        label = id_2_cls[int(class_id)]
        X = (np.array(segment[1:]).reshape((-1, 2))[:, 0] * 1440).astype(np.int32)
        Y = (np.array(segment[1:]).reshape((-1, 2))[:, 1] * 192).astype(np.int32)
        points = np.column_stack((X, Y))
        draw_polygon(image, points, label, color=(0, 0, 0), thickness=2)

    # Drawing the strip divisions
    for i in range(1, 6):
        cv2.line(image, (240*i, 0), (240*i+1, image.shape[0]), (0, 0, 0), 3)

    # Displaying the image
    cv2.imshow('Image with GT Vector', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    id_2_cls = {0: 'Broad leaf-1-2', 1: 'Grass leaf-1-2', 2: 'Amaranthus-1-2', 3: 'Soy-1-2'}
    image_path = r'test/images/219798.jpeg'
    x_ranges = [240, 480, 720, 960, 1200, 1440]
    GT_vec = get_GT_vector(219798)
    #draw_GT_vector(image_path,GT_vec)
    draw_GT_vector_with_seg(219798, GT_vec, id_2_cls)
