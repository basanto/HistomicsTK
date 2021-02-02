"""
Created on Mon Nov 23 9:00:07 2020
@author: Briana Santo

"""
import numpy as np
import skimage as sk
import scipy as sp
import cv2
from skimage import color
import matplotlib.pyplot as mplp

def get_wsi_mask(image,wsi_glom_mask,num_sections):
    wsi_mask = color.rgb2gray(image)
    wsi_mask = wsi_mask<(wsi_mask.mean())
    wsi_mask = sp.ndimage.morphology.binary_fill_holes(wsi_mask)
    wsi_mask = np.logical_and(wsi_mask,(np.invert(wsi_mask)))
    wsi_labels, num_labels = sp.ndimage.label(wsi_mask)
    wsi_props = sk.measure.regionprops(wsi_labels)
    euler_labels = np.empty([num_labels,2])
    areas_labels = np.empty([num_labels,2])
    for label in range(num_labels):
        euler_labels[label,0] = wsi_props[label].euler_number
        euler_labels[label,1] = (label+1)
        areas_labels[label,0] = wsi_props[label].area
        areas_labels[label,1] = (label+1)
    keep = np.where(euler_labels<0)[0]
    keep = areas_labels[keep,:]
    wsi_mask = np.zeros(wsi_mask.shape)
    wsi_mask[wsi_labels==(keep[:,1])] = 1
    wsi_mask = sp.ndimage.morphology.binary_fill_holes(wsi_mask)
    return(wsi_mask)
