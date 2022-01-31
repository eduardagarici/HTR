import numpy as np
import cv2
import h5py
import os
from skimage.filters import threshold_local

def processImage(imgPath, rows, cols):
    brightness = 0
    contrast = 50
    img = cv2.imread(imgPath)
    img = np.int16(img)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    T = threshold_local(img, 11, offset=10, method="gaussian")
    img = (img > T).astype("uint8") * 255

    # Increase line width
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    height, width = img.shape
    scaleX = width / cols
    scaleY = height / rows
    scale_factor = max(scaleX, scaleY)
    (newHeight,newWidth) =  (max(min(rows, int(height / scale_factor)), 1), max(min(cols, int(width / scale_factor)), 1)) # scale according to scale_factor (result at least 1 and at most wt or ht)
    img = cv2.resize(img, dsize = (newWidth, newHeight), interpolation = cv2.INTER_AREA)
    target = np.ones([rows, cols]) * 255
    dH = int((rows - newHeight) / 2)
    dW = int((cols - newWidth) / 2)
    target[dH:newHeight + dH, dW : newWidth + dW] = img
    img = target
    (mean, std) = cv2.meanStdDev(img)
    mean = mean[0][0]
    std = std[0][0]
    img -= mean
    img = img / std if std > 0 else img
    return img

# load images and scale to fixed height and width
def loadImagesAndScaleAndLabels(root_path, rows, cols, labels_path = None):

    images = list()
    for (dirpath, dirnames, filenames) in os.walk(root_path):
        images += [os.path.join(dirpath, file) for file in filenames]

    new_images = []
    labels = []
    if labels_path is not None:
        file = open(labels_path)
        lines = file.readlines()
        for image,line in zip(images,lines):
            img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
            if img is None :
                continue;
            img = processImage(image, rows, cols)
            new_images.append(img)
            labels.append(line.strip())
    else:
        for image in images:
            word_label = image[1 + image.rfind('-'):].split('.')[0]
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue;
            img = processImage(image, rows, cols)
            new_images.append(img)
            labels.append(word_label.strip())

    new_images = np.array(new_images)
    labels = np.array(labels, dtype = 'object')
    return new_images, labels

def loadImagesAndScaleAndLabelsForIam(root_path, rows, cols, labels_path, needed_images_path):

    images = list()
    for (dirpath, dirnames, filenames) in os.walk(root_path):
        images += [os.path.join(dirpath, file) for file in filenames]

    new_images = []
    labels = []

    file = open(labels_path)
    lines = file.readlines()

    needed_images_file = open(needed_images_path)
    needed_images_lines = needed_images_file.readlines()
    needed_images_lines.sort()
    needed_image_indice = 0

    for image,line in zip(images,lines):
        if image[image.rfind('\\')+1:image.rfind('.')].strip() == needed_images_lines[needed_image_indice].strip():

            needed_image_indice += 1
            img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
            if img is None :
                continue
            img = processImage(image, rows, cols)
            new_images.append(img)
            lineSplit = line.strip().split(' ')
            groundTruth = ' '.join(lineSplit[8:])
            label = groundTruth.replace('|',' ')
            labels.append(label)
            if (len(needed_images_lines) == needed_image_indice):
                break

    new_images = np.array(new_images)
    labels = np.array(labels, dtype = 'object')
    return new_images, labels

def saveFileH5(path, filename, data, is_label = False):
    link = path + '/' + filename + '.h5'
    h5f = h5py.File(link, 'w')
    if is_label == True :
        string_dt = h5py.special_dtype(vlen = str)
        h5f.create_dataset(filename, data=data, dtype=string_dt)
    else :
        h5f.create_dataset(filename, data = data)
    h5f.close()

def loadFileH5(path, filename):
    link = path + '/' + filename + '.h5'
    f = h5py.File(link, 'r')
    images = np.array(f[list(f.keys())[0]])
    f.close()
