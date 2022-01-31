import h5py
import numpy as np

def main():
    root = "G:/Licenta/Data/DateH5/IAM/"
    train = root + "labels_lines_train.h5"
    test = root + "labels_lines_test.h5"
    valid1 = root + "labels_lines_validation1.h5"
    valid2 = root + "labels_lines_validation2.h5"

    f = h5py.File(train, 'r')
    train = f.get(train[train.rfind('/') + 1:train.rfind('.')]).value
    chars = set()
    for label in train:
        chars = chars.union(set(list(label)))
    len(chars)

    '''
    f = h5py.File(test, 'r')
    test = f.get(test[test.rfind('/') + 1:test.rfind('.')]).value
    
    f = h5py.File(valid1 , 'r')
    valid1 = f.get(valid1[valid1.rfind('/') + 1:valid1.rfind('.')]).value
    
    f = h5py.File(valid2, 'r')
    
    valid2 = f.get(valid2[valid2.rfind('/') + 1:valid2.rfind('.')]).value
    
    valid = np.stack(valid1, valid2)
    '''
