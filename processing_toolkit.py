import SimpleITK as sitk
import numpy as np
import numpy.ma as ma
import sys
import os
from os import listdir
import glob
from os.path import isfile, join
import SimpleITK as sitk 
import itk
import numpy as np

def reorient_to_LPS(image):
    # Define the transformation needed for LPS orientation
    # This example assumes a flip along the x and y axes is necessary
    transform = sitk.AffineTransform(3)
    transform.SetMatrix([-1, 0, 0, 0, -1, 0, 0, 0, 1])

    # Define the resampling parameters
    reference_image = image
    interpolator = sitk.sitkLinear
    default_value = 0

    # Execute the resampling
    resampled_image = sitk.Resample(image, reference_image, transform, interpolator, default_value)

    # Update the direction to match LPS if necessary
    # Note: This step may require adjustment based on the specific orientation conversion
    resampled_image.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))

    return resampled_image

def reorient_to_LPS_GT(image):
    # Define the transformation needed for LPS orientation
    # This example assumes a flip along the x and y axes is necessary
    transform = sitk.AffineTransform(3)
    transform.SetMatrix([-1, 0, 0, 0, -1, 0, 0, 0, 1])

    # Define the resampling parameters
    reference_image = image
    interpolator = sitk.sitkNearestNeighbor
    default_value = 0

    # Execute the resampling
    resampled_image = sitk.Resample(image, reference_image, transform, interpolator, default_value)

    # Update the direction to match LPS if necessary
    # Note: This step may require adjustment based on the specific orientation conversion
    resampled_image.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))

    return resampled_image


def resample_and_reorient(input, ref):
    
    filter = sitk.ResampleImageFilter()
    filter.SetReferenceImage(ref)
    new_input_image = filter.Execute(input)
    new_input_image.CopyInformation(ref)

    new_input_image = reorient_to_LPS(new_input_image)

    return new_input_image