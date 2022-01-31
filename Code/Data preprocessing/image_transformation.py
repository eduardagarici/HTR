from math import ceil 
import numpy as np
import cv2
from scipy.ndimage import interpolation
from skimage.filters import threshold_local

width = 800
height = 64

class Result(object):
    def __init__(self, sum_alpha = 0.0, transform = None, size = (0,0)):
      self.sum_alpha = sum_alpha
      self.transform = transform
      self.size = size
    def __lt__(self, other) :
      return self.sum_alpha < other.sum_alpha

def deslant(img, bgColor = 0): # bgColor is the color with which is filled the empty space
  _,imgBW = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  alphaVals = np.array([-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
  sum_alpha = np.zeros(alphaVals.size)
  results = np.empty(alphaVals.size, dtype=object)
  for i in range(alphaVals.size):
   
      result = Result()
      alpha = alphaVals[i]
      shiftX = max(-alpha * imgBW.shape[0], 0.0)
      result.size = (imgBW.shape[1] + ceil(abs(alpha * imgBW.shape[0])), imgBW.shape[0])
      result.transform = np.zeros((2,3), dtype='float32')
      result.transform[0, 0] = 1
      result.transform[0, 1] = alpha
      result.transform[0, 2] = shiftX
      result.transform[1, 0] = 0;
      result.transform[1, 1] = 1;
      result.transform[1, 2] = 0;
    
      imgSheared = cv2.warpAffine(imgBW, result.transform, result.size, cv2.INTER_NEAREST)
      for x in range(imgSheared.shape[1]):
          fgIndices = []
          for y in range(imgSheared.shape[0]):
        
              if(imgSheared[y,x]):
                fgIndices.append(y)

          if(len(fgIndices) == 0):
              continue;

          h_alpha = len(fgIndices)
          delta_y_alpha = fgIndices[len(fgIndices) - 1] - fgIndices[0] + 1;
          if(h_alpha == delta_y_alpha):
              result.sum_alpha += h_alpha * h_alpha
      results[i] = result
  bestResult = max(results)
  img = cv2.warpAffine(img, bestResult.transform, bestResult.size, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,borderValue = (bgColor,bgColor,bgColor))
  return scale(img,height,width)

def scale(img, rows, cols):

    height, width = img.shape
    scaleX = width / cols
    scaleY = height / rows
    scale_factor = max(scaleX, scaleY)
    (newHeight, newWidth) = (max(min(rows, int(height / scale_factor)), 1), max(min(cols, int(width / scale_factor)),
                                                                                1))  # scale according to scale_factor (result at least 1 and at most wt or ht)
    img = cv2.resize(img, dsize=(newWidth, newHeight), interpolation=cv2.INTER_AREA)
    target = np.zeros([rows, cols])
    dH = int((rows - newHeight) / 2)
    dW = int((cols - newWidth) / 2)
    target[dH:newHeight + dH, dW: newWidth + dW] = img
    return target

def deskewOpenCV(image):

    gray = cv2.bitwise_not(image)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated = np.array(rotated, dtype=np.float32)
    (mean, std) = cv2.meanStdDev(rotated)
    mean = mean[0][0]
    std = std[0][0]
    rotated -= mean
    rotated = rotated / std if std > 0 else rotated

    return np.array(rotated)