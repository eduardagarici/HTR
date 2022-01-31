import os
import random
import numpy as np
import h5py
import Augmentor
import cv2

class Batch:

    def __init__(self, groundTexts, images):
        self.groundTexts = groundTexts
        self.images = np.stack(images, axis=0)


class DataLoader:
    root = 'drive/My Drive/Licenta/Data/dateH5/IAM/'
    normalImagesTrainPath = os.path.join(root, 'lines_train.h5')
    labelsTrainPath = os.path.join(root, 'labels_lines_train.h5')
    normalImagesValidation1Path = os.path.join(root, 'lines_validation1.h5')
    labelsValidation1Path = os.path.join(root, 'labels_lines_validation1.h5')
    normalImagesValidation2Path = os.path.join(root, 'lines_validation2.h5')
    labelsValidation2Path = os.path.join(root, 'labels_lines_validation2.h5')

    deslantedImagesTrainPath = os.path.join(root, 'deslanted_lines_train.h5')
    deslantedImagesValidation1Path = os.path.join(root, 'deslanted_lines_validation1.h5')
    deslantedImagesValidation2Path = os.path.join(root, 'deslanted_lines_validation2.h5')

    def __init__(self, batchSize, imageSize, maxTextLength,deslant,augment = False, rotate = False):
        self.dataAugmentation = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imageSize = imageSize
        self.maxTextLength = maxTextLength
        self.deslant = deslant
        self.rotate = rotate
        self.augment = augment

        if self.deslant:
            train = DataLoader.deslantedImagesTrainPath
            valid1 = DataLoader.deslantedImagesValidation1Path
            valid2 = DataLoader.deslantedImagesValidation2Path
        else:
            train = DataLoader.normalImagesTrainPath
            valid1 = DataLoader.normalImagesValidation1Path
            valid2 = DataLoader.normalImagesValidation2Path

        hf = h5py.File(train, 'r')
        self.trainImages = hf.get(train[train.rfind('/') + 1:train.rfind('.')]).value

        hf = h5py.File(valid1, 'r')
        self.validation1Images = hf.get(valid1[valid1.rfind('/') + 1:valid1.rfind('.')]).value

        hf = h5py.File(valid2, 'r')
        self.validation2Images = hf.get(valid2[valid2.rfind('/') + 1:valid2.rfind('.')]).value

        f = h5py.File(DataLoader.labelsTrainPath, 'r')
        self.trainLabels = np.array(f[list(f.keys())[0]])
        self.trainLabels = np.array([self.truncateLabel(label, self.maxTextLength) for label in self.trainLabels])

        f = h5py.File(DataLoader.labelsValidation1Path, 'r')
        self.validation1Labels = np.array(f[list(f.keys())[0]])

        f = h5py.File(DataLoader.labelsValidation2Path, 'r')
        self.validation2Labels = np.array(f[list(f.keys())[0]])

        chars = set()
        for label in self.trainLabels:
            chars = chars.union(set(list(label)))
        for label in self.validation1Labels:
            chars = chars.union(set(list(label)))
        for label in self.validation2Labels:
            chars = chars.union(set(list(label)))

        self.validationImages = np.append(self.validation1Images, self.validation2Images, axis=0)
        del self.validation1Images
        del self.validation2Images

        self.trainSamplesImages = np.append(self.trainImages, self.validationImages, axis = 0)

        self.validationLabels = np.append(self.validation1Labels, self.validation2Labels, axis=0)
        self.validationLabels = np.array(
            [self.truncateLabel(label, self.maxTextLength) for label in self.validationLabels])
        self.trainSamplesLabels = np.append(self.trainLabels, self.validationLabels, axis = 0)

        splitIndex = int(0.9 * len(self.trainSamplesImages))
        self.trainImages = self.trainSamplesImages[:splitIndex]
        self.trainLabels = self.trainSamplesLabels[:splitIndex]
        self.validationImages = self.trainSamplesImages[splitIndex:]
        self.validationLabels = self.trainSamplesLabels[splitIndex:]

        del self.trainSamplesImages
        self.charList = sorted(list(chars))

        if self.augment:
            self.augmentor = Augmentor.Pipeline()
            #self.augmentor.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=1)
            self.augmentor.rotate(probability=0.5, max_left_rotation=1, max_right_rotation=1)
            self.augmentor.skew_tilt(probability=0.5, magnitude=0.05)

    def truncateLabel(self, line, maxTextLenght):

        cost = 0
        for i in range(len(line)):
            if i != 0 and line[i] == line[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLenght:
                return line[:i]
        return line

    def trainSet(self):
        self.samples = [(image, label) for (image, label) in zip(self.trainImages, self.trainLabels)]
        self.dataAugmentation = False
        self.currIdx = 0
        random.shuffle(self.samples)

    def validationSet(self):
        self.samples = [(image, label) for (image, label) in zip(self.validationImages, self.validationLabels)]
        self.dataAugmentation = False
        self.currIdx = 0

    def getIteratorInfo(self):
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

    def hasNext(self):
        return self.currIdx + self.batchSize <= len(self.samples)

    def getNext(self):
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        groundTexts = [self.samples[i][1] for i in batchRange]
        images = [self.samples[i][0] for i in batchRange]
        if self.augment:
            generator = self.augmentor.keras_generator_from_array(images, groundTexts, batch_size= self.batchSize)
            images,groundTexts = next(generator)
            images = np.squeeze(images)
        if self.rotate:
            images = []
            for i in batchRange:
                (h, w) = self.samples[i][0].shape
                center = (w / 2, h / 2)
                scale = 1.0
                angle = (random.random() * 2 - 1) * 2
                M = cv2.getRotationMatrix2D(center, angle, scale)
                rotated = cv2.warpAffine(self.samples[i][0], M, (w, h))
                images.append(rotated)
        images = np.transpose(images, [0, 2, 1])

        self.currIdx += self.batchSize
        return Batch(groundTexts, images)
