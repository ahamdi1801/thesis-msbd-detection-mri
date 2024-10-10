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

def orientation_to_ras(img):
    

    # this function is implemented using ITK since SimpleITK has not implemented the filter "OrientImageFilter" yet
    # the ITK orientation system is from this blog post: https://itk.org/pipermail/insight-users/2017-May/054606.html
    # comparison ITK - simpleITK filters: https://itk.org/SimpleITKDoxygen/html/Filter_Coverage.html
    # see also: https://github.com/fedorov/lidc-idri-conversion/blob/master/seg/seg_converter.py

    # change image name
    img_sitk = img

    # get characteristics of simpleITK image
    size_in_sitk      = img_sitk.GetSize()
    spacing_in_sitk   = img_sitk.GetSpacing()
    origin_in_sitk    = img_sitk.GetOrigin()
    direction_in_sitk = img_sitk.GetDirection()

    # allocate ITK image (type and size)
    Dimension   = 3
    PixelType   = itk.F
    ImageTypeIn = itk.Image[PixelType, Dimension]
    img_itk = ImageTypeIn.New()
    sizeIn_itk = itk.Size[Dimension]()
    for i in range (0,Dimension):
        sizeIn_itk[i] = size_in_sitk[i]
    region = itk.ImageRegion[Dimension]()
    region.SetSize(sizeIn_itk)
    img_itk.SetRegions(region)
    img_itk.Allocate()

    # pass image from simpleITK to numpy
    img_py  = sitk.GetArrayFromImage(img_sitk)

    # pass image from numpy to ITK
    img_itk = itk.GetImageViewFromArray(img_py)

    # pass characteristics from simpleITK image to ITK image (except size, assigned in allocation)
    spacing_in_itk = itk.Vector[itk.F, Dimension]()
    for i in range (0,Dimension):
        spacing_in_itk[i] = spacing_in_sitk[i]
    img_itk.SetSpacing(spacing_in_itk)

    origin_in_itk  = itk.Point[itk.F, Dimension]()
    for i in range (0,Dimension):
        origin_in_itk[i]  = origin_in_sitk[i]
    img_itk.SetOrigin(origin_in_itk)

    # old way of assigning direction (until ITK 4.13)
#     direction_in_itk = itk.Matrix[itk.F,3,3]()
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(0,0,direction_in_sitk[0]) # r,c,value
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(0,1,direction_in_sitk[1])
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(0,2,direction_in_sitk[2])
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(1,0,direction_in_sitk[3]) # r,c,value
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(1,1,direction_in_sitk[4])
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(1,2,direction_in_sitk[5])
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(2,0,direction_in_sitk[6]) # r,c,value
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(2,1,direction_in_sitk[7])
#     direction_in_itk = img_itk.GetDirection().GetVnlMatrix().set(2,2,direction_in_sitk[8])

    direction_in_itk = np.eye(3)
    direction_in_itk[0][0] = direction_in_sitk[0]
    direction_in_itk[0][1] = direction_in_sitk[1]
    direction_in_itk[0][2] = direction_in_sitk[2]
    direction_in_itk[1][0] = direction_in_sitk[3]
    direction_in_itk[1][1] = direction_in_sitk[4]
    direction_in_itk[1][2] = direction_in_sitk[5]
    direction_in_itk[2][0] = direction_in_sitk[6]
    direction_in_itk[2][1] = direction_in_sitk[7]
    direction_in_itk[2][2] = direction_in_sitk[8]
    img_itk.SetDirection(itk.matrix_from_array(direction_in_itk))

    # make sure image is float for the orientation filter (GetImageViewFromArray sets it to unsigned char)
    ImageTypeIn_afterPy = type(img_itk)
    ImageTypeOut        = itk.Image[itk.F, 3]
    CastFilterType      = itk.CastImageFilter[ImageTypeIn_afterPy, ImageTypeOut]
    castFilter          = CastFilterType.New()
    castFilter.SetInput(img_itk)
    castFilter.Update()
    img_itk             = castFilter.GetOutput()

    # define ITK orientation system  (from the blog post: https://itk.org/pipermail/insight-users/2017-May/054606.html)
    ITK_COORDINATE_UNKNOWN   = 0
    ITK_COORDINATE_Right     = 2
    ITK_COORDINATE_Left      = 3
    ITK_COORDINATE_Posterior = 4
    ITK_COORDINATE_Anterior  = 5
    ITK_COORDINATE_Inferior  = 8
    ITK_COORDINATE_Superior  = 9
    ITK_COORDINATE_PrimaryMinor   = 0
    ITK_COORDINATE_SecondaryMinor = 8
    ITK_COORDINATE_TertiaryMinor  = 16

    ###below Define ITK orientation for RAS
    ITK_COORDINATE_ORIENTATION = (ITK_COORDINATE_Right    << ITK_COORDINATE_PrimaryMinor) \
                                    + (ITK_COORDINATE_Anterior << ITK_COORDINATE_SecondaryMinor) \
                                    + (ITK_COORDINATE_Superior << ITK_COORDINATE_TertiaryMinor)

    ##below Define ITK orientation for RAI
#    ITK_COORDINATE_ORIENTATION = ( ITK_COORDINATE_Right    << ITK_COORDINATE_PrimaryMinor ) \
 #                                  + ( ITK_COORDINATE_Anterior << #ITK_COORDINATE_SecondaryMinor ) \
 #                                  + ( ITK_COORDINATE_Inferior << ITK_COORDINATE_TertiaryMinor )
    
    # Change orientation to RAS
    OrientType = itk.OrientImageFilter[ImageTypeOut,ImageTypeOut]
    filter = OrientType.New()
    filter.UseImageDirectionOn()
    filter.SetDesiredCoordinateOrientation(ITK_COORDINATE_ORIENTATION)  # Changed to RAS
    filter.SetInput(img_itk)
    filter.Update()
    img_itk = filter.GetOutput()

    

    ##above Define ITK orientation for RAI
    
    # get characteristics of ITK image
    spacing_out_itk   = img_itk.GetSpacing()
    origin_out_itk    = img_itk.GetOrigin()

    # pass image from itk to numpy
    img_py = itk.GetArrayViewFromImage(img_itk)

    # pass image from numpy to simpleitk
    img = sitk.GetImageFromArray(img_py)

    # pass characteristics from ITK image to simpleITK image (except size, implicitely passed)
    spacing = []
    for i in range (0, Dimension):
        spacing.append(spacing_out_itk[i])
    img.SetSpacing(spacing)

    origin = []
    for i in range (0, Dimension):
        origin.append(origin_out_itk[i])
    img.SetOrigin(origin)

    direction = []
    direction.append(1.0)
    direction.append(0.0)
    direction.append(0.0)
    direction.append(0.0)
    print('chqnged version2')
    direction.append(-1.0)
    direction.append(0.0)
    direction.append(0.0)
    direction.append(0.0)
    direction.append(1.0)
    img.SetDirection(direction)

    return img

def orientation_to_lps(img_sitk):
    # Extracting the initial properties from the SimpleITK image
    size_in_sitk = img_sitk.GetSize()
    spacing_in_sitk = img_sitk.GetSpacing()
    origin_in_sitk = img_sitk.GetOrigin()
    direction_in_sitk = img_sitk.GetDirection()

    # Convert SimpleITK image to ITK image
    Dimension = 3
    PixelType = itk.F
    ImageTypeIn = itk.Image[PixelType, Dimension]

    img_itk = itk.GetImageViewFromArray(sitk.GetArrayFromImage(img_sitk))
    img_itk.SetSpacing(spacing_in_sitk)
    img_itk.SetOrigin(origin_in_sitk)
    img_itk.SetDirection(itk.matrix_from_array(np.array(direction_in_sitk).reshape(Dimension, Dimension)))

    # Casting to ensure image is float for orientation filter
    ImageTypeOut = itk.Image[itk.F, Dimension]
    castFilter = itk.CastImageFilter[ImageTypeIn, ImageTypeOut].New()
    castFilter.SetInput(img_itk)
    castFilter.Update()
    img_itk = castFilter.GetOutput()

    ITK_COORDINATE_UNKNOWN   = 0
    ITK_COORDINATE_Right     = 2
    ITK_COORDINATE_Left      = 3
    ITK_COORDINATE_Posterior = 4
    ITK_COORDINATE_Anterior  = 5
    ITK_COORDINATE_Inferior  = 8
    ITK_COORDINATE_Superior  = 9
    ITK_COORDINATE_PrimaryMinor   = 0
    ITK_COORDINATE_SecondaryMinor = 8
    ITK_COORDINATE_TertiaryMinor  = 16
    # Since ITK uses LPS by default, we don't need to explicitly set LPS orientation
    # This is for educational purposes: showing how to use OrientImageFilter without changing orientation
    ITK_COORDINATE_ORIENTATION_LPS = (ITK_COORDINATE_Left << ITK_COORDINATE_PrimaryMinor) + \
                                     (ITK_COORDINATE_Posterior << ITK_COORDINATE_SecondaryMinor) + \
                                     (ITK_COORDINATE_Superior << ITK_COORDINATE_TertiaryMinor)

    # This operation is redundant for setting LPS in ITK but included for completeness
    OrientFilter = itk.OrientImageFilter[ImageTypeOut, ImageTypeOut].New()
    OrientFilter.SetInput(img_itk)
    OrientFilter.UseImageDirectionOn()
    OrientFilter.SetDesiredCoordinateOrientation(ITK_COORDINATE_ORIENTATION_LPS)
    OrientFilter.Update()
    img_itk = OrientFilter.GetOutput()

    # Convert ITK image back to SimpleITK
    img_lps = sitk.GetImageFromArray(itk.GetArrayFromImage(img_itk))
    img_lps.SetSpacing(list(img_itk.GetSpacing()))
    img_lps.SetOrigin(list(img_itk.GetOrigin()))
    direction_array = itk.GetArrayFromVnlMatrix(img_itk.GetDirection().GetVnlMatrix().as_matrix())
    img_lps.SetDirection(direction_array.flatten())

    print("new version")

    return img_lps
    
def orientation_to_lps2(img_sitk):
    new_direction=[-1,0,0,0,-1,0,0,0,1]
    img_sitk.SetDirection(new_direction)
    return img_sitk


# def reorient_image_to_LPS(image):
#     # Define the affine transformation for flipping and swapping axes
#     # This matrix represents flipping the y-axis and swapping y and z axes
#     transform_matrix = [-1, 0, 0, 0, 0, 1, 0, -1, 0]
#     transform = sitk.AffineTransform(3)
#     transform.SetMatrix(transform_matrix)
    
#     # The affine transform will move the origin, so we set it back to the original image's origin
#     transform.SetTranslation([0, 0, 0])
    
#     # Set up the resampling parameters
#     interpolator = sitk.sitkLinear
#     output_image = sitk.Image(image.GetSize(), image.GetPixelID())
#     output_image.SetSpacing(image.GetSpacing())
    
#     # Adjust the output direction to match the LPS orientation
#     output_image.SetDirection(transform_matrix)
    
#     # The origin and spacing remain the same, but we need to adjust the direction
#     output_image.SetOrigin(image.GetOrigin())
    
#     # Resample the original image with the specified transformation
#     resampled_image = sitk.Resample(image, output_image, transform, interpolator, 0.0, image.GetPixelID())

#     return resampled_image

# def reorient_image_to_LPS_GT(image):
#     # Define the affine transformation for flipping and swapping axes
#     # This matrix represents flipping the y-axis and swapping y and z axes
#     filter=sitk.MinimumMaximumImageFilter()
#     filter.Execute(image)
#     print(filter.GetMaximum()) 

#     transform_matrix = [-1, 0, 0, 0, 0, 1, 0, -1, 0]
#     transform = sitk.AffineTransform(3)
#     transform.SetMatrix(transform_matrix)
#     resample=sitk.ResampleImageFilter()
    
#     # The affine transform will move the origin, so we set it back to the original image's origin
#     transform.SetTranslation([0, 0, 0])
#     resample.SetTransform(transform)
#     interpolator = sitk.sitkLinear
#     resample.SetInterpolator(interpolator)
#     resample.SetUseNearestNeighborExtrapolator(True)
#     # Set up the resampling parameters
    
#     output_image = sitk.Image(image.GetSize(), image.GetPixelID())
#     output_image.SetSpacing(image.GetSpacing())
    
#     # Adjust the output direction to match the LPS orientation
#     output_image.SetDirection(transform_matrix)
    
#     # The origin and spacing remain the same, but we need to adjust the direction
#     output_image.SetOrigin(image.GetOrigin())
#     resample.SetReferenceImage(output_image)
#     resampled_image=resample.Execute(image)
#     # Resample the original image with the specified transformation
#    # resampled_image = sitk.Resample(image, output_image, transform, interpolator, 0.0, image.GetPixelID(),True)

#     filter.Execute(resampled_image)
#     print(filter.GetMaximum()) 
#     return resampled_image

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




def RAI(patient):
    for image_name in glob.glob(patient+'/*.mhd'):
        image=orientation_to_rai(sitk.ReadImage(image_name))
        writer = sitk.ImageFileWriter()
        writer.SetFileName(image_name)
        writer.Execute(image)

def change_direction(list_of_images):
    for image_name in list_of_images:
        image=sitk.ReadImage(image_name)
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator = sitk.sitkBSpline

        resample.SetReferenceImage(image)
        resample.SetOutputDirection([ 1. , 0.,  0., -0., -0.,  1.,  0., -1., -0.])
        new_image=resample.Execute(image)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(image_name)
        writer.Execute(new_image)