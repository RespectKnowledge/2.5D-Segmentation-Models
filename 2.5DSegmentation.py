# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 08:54:45 2019

@author: Abdul Qayyum

"""
# reference github used for image segmentation and dataset preprtion
#http://aapmchallenges.cloudapp.net/forums/3/4/
#http://aapmchallenges.cloudapp.net/forums/3/2/
#https://github.com/wxkkk/OARs_Seg
#https://github.com/wxkkk?after=Y3Vyc29yOnYyOpK5MjAxOS0wNy0wM1QwNTozMTozMCswMjowMM4KhXT6&direction=desc&sort=updated&tab=stars
#https://github.com/wxkkk?direction=desc&sort=updated&tab=stars
#https://github.com/Mohitasudani/Segthor19-using-ResU-net
#https://github.com/yasharmaster/image-segmentation-demo
import os, sys, glob

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

ROI_ORDER = ['SpinalCord', 'Lung_R', 'Lung_L', 'Heart', 'Esophagus']


def read_data(path, train_flag, floder_flag, out_path):
    '''
     read training set and validation set according to directory structure
    :param path:
    :param train_flag: future work for semi-supervised method to generate new masks
    :param floder_flag: output folder
    :param out_path: output path
    :return:
    '''
    for root, dirs, file in os.walk(path):
        for i in dirs:
            # get the number of current path
            dicoms = glob.glob(os.path.join(root + '/' + i, '*.dcm'))

            # read RT structure files
            if train_flag == 'train_set' and 'simplified' in os.path.split(i)[1]:
                rt_dicom = pydicom.dcmread(dicoms[0])
                contours = get_contours(rt_dicom)

            # read dicoms
            elif len(dicoms) > 0:
                scans = [None] * len(dicoms)
                for i in range(len(dicoms)):
                    scans[i] = pydicom.dcmread(dicoms[i])
                    # print(scans[i].pixel_array.shape)
                    # plt.imshow(scans[i].pixel_array, cmap=plt.cm.bone)
                    # plt.show()

                # ['ImagePositionPatient', 'PatientPosition', 'PositionReferenceIndicator']
                print(scans[i].dir('position'))
                # print(scans[i].ImagePositionPatient)
                # sort by z
                scans.sort(key=lambda z: float(z.ImagePositionPatient[2]))
                #
                images = np.stack([s.pixel_array for s in scans])
                images = images.astype(np.int16)
                print(images.shape)

                # HU
                for s_num in range(len(scans)):
                    intercept = scans[s_num].RescaleIntercept
                    slope = scans[s_num].RescaleSlope
                    # print('inter:', intercept, 'slope', slope)

                    if slope != 1:
                        images[s_num] = slope * images[s_num].astype(np.float64)
                        images[s_num] = images[s_num].astype(np.int16)

                    images[s_num] += np.int16(intercept)
                    # plot HU for every image
                    # plt.hist(images[s_num].flatten(), bins=80, color='c')
                    # plt.show()

    # normalization HU
    images = normalization_hu(images)
    # plt.hist(images[s_num].flatten(), bins=80, color='c')
    # plt.show()

    if train_flag == 'train_set':

        # save images as npy
        for img in range(0, len(images)):
            total_imgs = []
            folder = os.path.exists(out_path + '/train_' + str(floder_flag) + '/images')
            if not folder:
                os.mkdir(out_path + '/train_' + str(floder_flag))
                os.mkdir(out_path + '/train_' + str(floder_flag) + '/images')

            if img == 0 or img == len(images) - 1:
                continue
            # print(img)
            # 2.5D data, using adjacent 3 images
            cur_img = images[img - 1:img + 2, :, :].astype('uint8')
            np.save(out_path + '/train_' + str(floder_flag) + '/images/' + str(img) + '_image.npy', cur_img)


        masks = get_masks(contours, images.shape, scans)

        # save masks as npy
        for msk in range(0, len(masks)):
            # save as png
            # labels_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
            # print('base name', os.path.basename(root))
            folder = os.path.exists(out_path + '/train_' + str(floder_flag) + "/masks")
            if not folder:
                os.mkdir(out_path + '/train_' + str(floder_flag) + "/masks")
            if msk == 0 or msk == len(images) - 1:
                continue
            # cur_mask = masks[msk - 1: msk + 2, :, :].astype('uint8')
            # cur_mask = masks[msk, :, :].astype('uint8')
            np.save(out_path + '/train_' + str(floder_flag) + '/masks/' + str(msk) + '_mask.npy', masks[msk])

        return images, masks

    # future work for semi-supervised learning
    # elif train_flag == 'semi_supervised_set':
    #     # save images as png
    #     # need to be un comment
    #     for img in range(0, len(images)):
    #         # images_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
    #         folder = os.path.exists(out_path + '/test_' + str(floder_flag) + "/images")
    #         if not folder:
    #             os.mkdir(out_path + "/images")
    #         else:
    #             np.save(out_path + '/train_' + str(floder_flag) + '/images/' + str(img) + '_image.npy', cur_img)
    #     return images


# some code imported from http://aapmchallenges.cloudapp.net/forums/3/2/
def get_contours(structure):
    '''
    get structure and reference information from RT Structure DICOM file
    :param structure: RT Structure file
    :return: contour
    '''
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
    return contours


def get_masks(contours, shape, scans):
    '''
    convert RT Structure to label map mask
    :param contours
    :param shape:
    :param scans:
    :return:
    '''
    z = [np.around(s.ImagePositionPatient[2], 1) for s in scans]
    pos_row = scans[0].ImagePositionPatient[1]
    # print(scans[0].ImagePositionPatient)
    spacing_row = scans[0].PixelSpacing[1]
    pos_column = scans[0].ImagePositionPatient[0]
    spacing_column = scans[0].PixelSpacing[0]

    mask = np.zeros(shape, dtype=np.float32)
    for con in contours:
        num = ROI_ORDER.index(con['name']) + 1
        for c in con['contours']:
            nodes = np.array(c).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = z.index(np.around(nodes[0, 2], 1))
            # print('z_index: ', z_index)
            r = (nodes[:, 1] - pos_row) / spacing_row
            c = (nodes[:, 0] - pos_column) / spacing_column
            rr, cc = polygon(r, c)
            mask[z_index, rr, cc] = int(num)
    # print('mask shape: ', mask[0], 'num shape', num)
    return mask
# import code end


def normalization_hu(images):
    '''
    normalization to 0-255. and CT value from -1000 to 800 HU
    :param images:
    :return: normalized image
    '''
    MIN = -1000
    MAX = 800
    images = (images - MIN) / (MAX - MIN)
    images[images > 1] = 1.
    images[images < 0] = 0.
    return images * 255


def plot_ct_scan(scan):
    '''
    plot a few more images of the slices
    :param scan:
    :return:
    '''
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(30, 30))  # set 7 for all images
    # print('shape[0]', scan.shape[0])
    for i in range(0, scan.shape[0], 5):
        for j in range(4):
            # plots[int(i / 20), j].axis('off')
            plots[int(i / 20), j].imshow(scan[i], cmap=plt.cm.bone)
    plt.show()


if __name__ == '__main__':
    # set the path
    total_data_path = ""
    total_data_path_val = ""

    images_output_path = ""
    images_output_path_val = ""

    cases = [os.path.join(total_data_path_val, name)
             for name in sorted(os.listdir(total_data_path_val)) if
             os.path.isdir(os.path.join(total_data_path_val, name))]
    print('Patient number: ', len(cases))

    # # train sets
    for c in cases:
        # for folder_flag in range(3):
        print('c index:', cases.index(c))
        folder_flag = cases.index(c)
        print('C: ', os.path.basename(os.path.dirname(c)))
        images, masks = read_data(c, 'train_set', folder_flag, images_output_path_val)

    plot_ct_scan(images)
#%% Tested dataset  code ######################## Tarining and Testing AAPM 2017 dataset convert into numpy
import os, sys, glob

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

ROI_ORDER = ['SpinalCord', 'Lung_R', 'Lung_L', 'Heart', 'Esophagus']


def read_data(path, train_flag, floder_flag, out_path):
    '''
     read training set and validation set according to directory structure
    :param path:
    :param train_flag: future work for semi-supervised method to generate new masks
    :param floder_flag: output folder
    :param out_path: output path
    :return:
    '''
    for root, dirs, file in os.walk(path):
        for i in dirs:
            # get the number of current path
            dicoms = glob.glob(os.path.join(root + '\\' + i, '*.dcm'))

            # read RT structure files
            if train_flag == 'train_set' and 'simplified' in os.path.split(i)[1]:
                rt_dicom = pydicom.dcmread(dicoms[0])
                contours = get_contours(rt_dicom)

            # read dicoms
            elif len(dicoms) > 0:
                scans = [None] * len(dicoms)
                for i in range(len(dicoms)):
                    scans[i] = pydicom.dcmread(dicoms[i])
                    # print(scans[i].pixel_array.shape)
                    # plt.imshow(scans[i].pixel_array, cmap=plt.cm.bone)
                    # plt.show()

                # ['ImagePositionPatient', 'PatientPosition', 'PositionReferenceIndicator']
                print(scans[i].dir('position'))
                # print(scans[i].ImagePositionPatient)
                # sort by z
                scans.sort(key=lambda z: float(z.ImagePositionPatient[2]))
                #
                images = np.stack([s.pixel_array for s in scans])
                images = images.astype(np.int16)
                print(images.shape)

                # HU
                for s_num in range(len(scans)):
                    intercept = scans[s_num].RescaleIntercept
                    slope = scans[s_num].RescaleSlope
                    # print('inter:', intercept, 'slope', slope)

                    if slope != 1:
                        images[s_num] = slope * images[s_num].astype(np.float64)
                        images[s_num] = images[s_num].astype(np.int16)

                    images[s_num] += np.int16(intercept)
                    # plot HU for every image
                    # plt.hist(images[s_num].flatten(), bins=80, color='c')
                    # plt.show()

    # normalization HU
    images = normalization_hu(images)
    # plt.hist(images[s_num].flatten(), bins=80, color='c')
    # plt.show()

    if train_flag == 'train_set':

        # save images as npy
        for img in range(0, len(images)):
            total_imgs = []
            folder = os.path.exists(out_path + '/train_' + str(floder_flag) + '/images')
            if not folder:
                os.mkdir(out_path + '/train_' + str(floder_flag))
                os.mkdir(out_path + '/train_' + str(floder_flag) + '/images')

            if img == 0 or img == len(images) - 1:
                continue
            # print(img)
            # 2.5D data, using adjacent 3 images
            cur_img = images[img - 1:img + 2, :, :].astype('uint8')
            np.save(out_path + '/train_' + str(floder_flag) + '/images/' + str(img) + '_image.npy', cur_img)


        masks = get_masks(contours, images.shape, scans)

        # save masks as npy
        for msk in range(0, len(masks)):
            # save as png
            # labels_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
            # print('base name', os.path.basename(root))
            folder = os.path.exists(out_path + '/train_' + str(floder_flag) + "/masks")
            if not folder:
                os.mkdir(out_path + '/train_' + str(floder_flag) + "/masks")
            if msk == 0 or msk == len(images) - 1:
                continue
            # cur_mask = masks[msk - 1: msk + 2, :, :].astype('uint8')
            # cur_mask = masks[msk, :, :].astype('uint8')
            np.save(out_path + '/train_' + str(floder_flag) + '/masks/' + str(msk) + '_mask.npy', masks[msk])

        return images, masks

    # future work for semi-supervised learning
    # elif train_flag == 'semi_supervised_set':
    #     # save images as png
    #     # need to be un comment
    #     for img in range(0, len(images)):
    #         # images_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
    #         folder = os.path.exists(out_path + '/test_' + str(floder_flag) + "/images")
    #         if not folder:
    #             os.mkdir(out_path + "/images")
    #         else:
    #             np.save(out_path + '/train_' + str(floder_flag) + '/images/' + str(img) + '_image.npy', cur_img)
    #     return images


# some code imported from http://aapmchallenges.cloudapp.net/forums/3/2/
def get_contours(structure):
    '''
    get structure and reference information from RT Structure DICOM file
    :param structure: RT Structure file
    :return: contour
    '''
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
    return contours


def get_masks(contours, shape, scans):
    '''
    convert RT Structure to label map mask
    :param contours
    :param shape:
    :param scans:
    :return:
    '''
    z = [np.around(s.ImagePositionPatient[2], 1) for s in scans]
    pos_row = scans[0].ImagePositionPatient[1]
    # print(scans[0].ImagePositionPatient)
    spacing_row = scans[0].PixelSpacing[1]
    pos_column = scans[0].ImagePositionPatient[0]
    spacing_column = scans[0].PixelSpacing[0]

    mask = np.zeros(shape, dtype=np.float32)
    for con in contours:
        num = ROI_ORDER.index(con['name']) + 1
        for c in con['contours']:
            nodes = np.array(c).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = z.index(np.around(nodes[0, 2], 1))
            # print('z_index: ', z_index)
            r = (nodes[:, 1] - pos_row) / spacing_row
            c = (nodes[:, 0] - pos_column) / spacing_column
            rr, cc = polygon(r, c)
            mask[z_index, rr, cc] = int(num)
    # print('mask shape: ', mask[0], 'num shape', num)
    return mask
# import code end


def normalization_hu(images):
    '''
    normalization to 0-255. and CT value from -1000 to 800 HU
    :param images:
    :return: normalized image
    '''
    MIN = -1000
    MAX = 800
    images = (images - MIN) / (MAX - MIN)
    images[images > 1] = 1.
    images[images < 0] = 0.
    return images * 255


def plot_ct_scan(scan):
    '''
    plot a few more images of the slices
    :param scan:
    :return:
    '''
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(30, 30))  # set 7 for all images
    # print('shape[0]', scan.shape[0])
    for i in range(0, scan.shape[0], 5):
        for j in range(4):
            # plots[int(i / 20), j].axis('off')
            plots[int(i / 20), j].imshow(scan[i], cmap=plt.cm.bone)
    plt.show()


if __name__ == '__main__':
    # set the path
    total_data_path = ""
    total_data_path_val = "D:\\Newdatasetandcodes\\OARnewdataset\\testmodify\\Test"

    images_output_path = ""
    images_output_path_val = "D:\\Newdatasetandcodes\\OARnewdataset\\Testoutput"

    cases = [os.path.join(total_data_path_val, name)
             for name in sorted(os.listdir(total_data_path_val)) if
             os.path.isdir(os.path.join(total_data_path_val, name))]
    print('Patient number: ', len(cases))

    # # train sets
    for c in cases:
        # for folder_flag in range(3):
        print('c index:', cases.index(c))
        folder_flag = cases.index(c)
        print('C: ', os.path.basename(os.path.dirname(c)))
        images, masks = read_data(c, 'train_set', folder_flag, images_output_path_val)

    plot_ct_scan(images)
#%% test modified
import os, sys, glob

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

ROI_ORDER = ['SpinalCord', 'Lung_R', 'Lung_L', 'Heart', 'Esophagus']


def read_data(path, train_flag, floder_flag, out_path):
    '''
     read training set and validation set according to directory structure
    :param path:
    :param train_flag: future work for semi-supervised method to generate new masks
    :param floder_flag: output folder
    :param out_path: output path
    :return:
    '''
    for root, dirs, file in os.walk(path):
        for i in dirs:
            # get the number of current path
            dicoms = glob.glob(os.path.join(root + '\\' + i, '*.dcm'))

            # read RT structure files
            if train_flag == 'train_set' and 'simplified' in os.path.split(i)[1]:
                rt_dicom = pydicom.dcmread(dicoms[0])
                contours = get_contours(rt_dicom)

            # read dicoms
            elif len(dicoms) > 0:
                scans = [None] * len(dicoms)
                for i in range(len(dicoms)):
                    scans[i] = pydicom.dcmread(dicoms[i])
                    # print(scans[i].pixel_array.shape)
                    # plt.imshow(scans[i].pixel_array, cmap=plt.cm.bone)
                    # plt.show()

                # ['ImagePositionPatient', 'PatientPosition', 'PositionReferenceIndicator']
                print(scans[i].dir('position'))
                # print(scans[i].ImagePositionPatient)
                # sort by z
                scans.sort(key=lambda z: float(z.ImagePositionPatient[2]))
                #
                images = np.stack([s.pixel_array for s in scans])
                images = images.astype(np.int16)
                print(images.shape)

                # HU
                for s_num in range(len(scans)):
                    intercept = scans[s_num].RescaleIntercept
                    slope = scans[s_num].RescaleSlope
                    # print('inter:', intercept, 'slope', slope)

                    if slope != 1:
                        images[s_num] = slope * images[s_num].astype(np.float64)
                        images[s_num] = images[s_num].astype(np.int16)

                    images[s_num] += np.int16(intercept)
                    # plot HU for every image
                    # plt.hist(images[s_num].flatten(), bins=80, color='c')
                    # plt.show()

    # normalization HU
    images1 = normalization_hu(images)
    images11=np.swapaxes(images1,0,2)
    # plt.hist(images[s_num].flatten(), bins=80, color='c')
    # plt.show()

    if train_flag == 'train_set':

        # save images as npy
        for img in range(0, len(images1)):
            total_imgs = []
            folder = os.path.exists(out_path + '/train_' + str(floder_flag) + '/images')
            if not folder:
                os.mkdir(out_path + '/train_' + str(floder_flag))
                os.mkdir(out_path + '/train_' + str(floder_flag) + '/images')

#            if img == 0 or img == len(images) - 1:
#                continue
            # print(img)
            # 2.5D data, using adjacent 3 images
            #cur_img = images[:,:,img - 1:img + 2].astype('uint8')
            cur_img = images11[img].astype('uint8')
            np.save(out_path + '/train_' + str(floder_flag) + '/images/' + str(img) + '_image.npy', cur_img)


        masks = get_masks(contours, images1.shape, scans)
        #masks=np.swapaxes(masks1,0,2)

        # save masks as npy
        for msk in range(0, len(masks)):
            # save as png
            # labels_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
            # print('base name', os.path.basename(root))
            folder = os.path.exists(out_path + '/train_' + str(floder_flag) + "/masks")
            if not folder:
                os.mkdir(out_path + '/train_' + str(floder_flag) + "/masks")
#            if msk == 0 or msk == len(images) - 1:
#                continue
            # cur_mask = masks[msk - 1: msk + 2, :, :].astype('uint8')
            # cur_mask = masks[msk, :, :].astype('uint8')
            np.save(out_path + '/train_' + str(floder_flag) + '/masks/' + str(msk) + '_mask.npy', masks[msk])

        return images, masks

    # future work for semi-supervised learning
    # elif train_flag == 'semi_supervised_set':
    #     # save images as png
    #     # need to be un comment
    #     for img in range(0, len(images)):
    #         # images_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
    #         folder = os.path.exists(out_path + '/test_' + str(floder_flag) + "/images")
    #         if not folder:
    #             os.mkdir(out_path + "/images")
    #         else:
    #             np.save(out_path + '/train_' + str(floder_flag) + '/images/' + str(img) + '_image.npy', cur_img)
    #     return images


# some code imported from http://aapmchallenges.cloudapp.net/forums/3/2/
def get_contours(structure):
    '''
    get structure and reference information from RT Structure DICOM file
    :param structure: RT Structure file
    :return: contour
    '''
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
    return contours


def get_masks(contours, shape, scans):
    '''
    convert RT Structure to label map mask
    :param contours
    :param shape:
    :param scans:
    :return:
    '''
    z = [np.around(s.ImagePositionPatient[2], 1) for s in scans]
    pos_row = scans[0].ImagePositionPatient[1]
    # print(scans[0].ImagePositionPatient)
    spacing_row = scans[0].PixelSpacing[1]
    pos_column = scans[0].ImagePositionPatient[0]
    spacing_column = scans[0].PixelSpacing[0]

    mask = np.zeros(shape, dtype=np.float32)
    for con in contours:
        num = ROI_ORDER.index(con['name']) + 1
        for c in con['contours']:
            nodes = np.array(c).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = z.index(np.around(nodes[0, 2], 1))
            # print('z_index: ', z_index)
            r = (nodes[:, 1] - pos_row) / spacing_row
            c = (nodes[:, 0] - pos_column) / spacing_column
            rr, cc = polygon(r, c)
            mask[z_index, rr, cc] = int(num)
    # print('mask shape: ', mask[0], 'num shape', num)
    return mask
# import code end


def normalization_hu(images):
    '''
    normalization to 0-255. and CT value from -1000 to 800 HU
    :param images:
    :return: normalized image
    '''
    MIN = -1000
    MAX = 800
    images = (images - MIN) / (MAX - MIN)
    images[images > 1] = 1.
    images[images < 0] = 0.
    return images * 255


def plot_ct_scan(scan):
    '''
    plot a few more images of the slices
    :param scan:
    :return:
    '''
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(30, 30))  # set 7 for all images
    # print('shape[0]', scan.shape[0])
    for i in range(0, scan.shape[0], 5):
        for j in range(4):
            # plots[int(i / 20), j].axis('off')
            plots[int(i / 20), j].imshow(scan[i], cmap=plt.cm.bone)
    plt.show()


if __name__ == '__main__':
    # set the path
    total_data_path = ""
    total_data_path_val = "D:\\Newdatasetandcodes\\OARnewdataset\\testmodify\\Test"

    images_output_path = ""
    images_output_path_val = "D:\\Newdatasetandcodes\\OARnewdataset\\Testoutput111"

    cases = [os.path.join(total_data_path_val, name)
             for name in sorted(os.listdir(total_data_path_val)) if
             os.path.isdir(os.path.join(total_data_path_val, name))]
    print('Patient number: ', len(cases))

    # # train sets
    for c in cases:
        # for folder_flag in range(3):
        print('c index:', cases.index(c))
        folder_flag = cases.index(c)
        print('C: ', os.path.basename(os.path.dirname(c)))
        images, masks = read_data(c, 'train_set', folder_flag, images_output_path_val)

    plot_ct_scan(images)    

#%%  convert images numpy into png based on dataset 2017 aapm
import natsort
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
from tqdm import tqdm_notebook, tnrange
import sys
import os
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
#print(os.listdir(path))

def truncated_range(img):
    max_hu = 384
    min_hu = -384
    img[np.where(img > max_hu)] = max_hu
    img[np.where(img < min_hu)] = min_hu
    return (img - min_hu) / (max_hu - min_hu) * 255. 
path1='D:\\Newdatasetandcodes\\OARnewdataset\\output'
directory_list=sorted(os.listdir(path1))
saveslicesimage='D:\\Newdatasetandcodes\\OARnewdataset\\2017aapmimages'
#trainmasks='D:\\Newdatasetandcodes\\OARnewdataset\\trainmasks'
d=1
for i in natsort.natsorted(directory_list,reverse=False):
    p=os.path.join(path1,i)
    #print(p)
    imageid = next(os.walk(p+"\\"+"images"))
    images=imageid[2]
    #print(images)
#    maskid = next(os.walk(p+"\\"+"masks"))
#    masks=maskid[2]
    #ii.split('.')[0]
    for ii in natsort.natsorted(images,reverse=False):
        print(os.path.join(p,ii))
        img = np.load(os.path.join(p+"\\"+"images",ii)).transpose(1, 2, 0)
        imagef = img[:,:,:]
        #stacked_img = np.stack((img1,)*3, axis=-1)
        imagef= resize(imagef, (512, 512, 3), mode='constant', preserve_range=True)
        #img=truncated_range(imagef)
        filename = "%d.png"%d
        cv2.imwrite(os.path.join(saveslicesimage,str(i)+'_'+str(ii.split('.')[0]+'.png')),imagef) # save images
        d+=1 
    print(i)
#%% convert mask 2017 aapm dataset from numpy to png
import natsort
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
from tqdm import tqdm_notebook, tnrange
import sys
import os
#print(os.listdir(path))

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from keras import metrics
from keras.utils import np_utils, plot_model
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras.layers.core import Lambda, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
path1='D:\\Newdatasetandcodes\\OARnewdataset\\output'
directory_list=sorted(os.listdir(path1))
trainmasks='D:\\Newdatasetandcodes\\OARnewdataset\\2017aapmmasks'
d=1
for i in natsort.natsorted(directory_list,reverse=False):
    p=os.path.join(path1,i)
    #print(p)
    maskid = next(os.walk(p+"\\"+"masks"))
    masks=maskid[2]
    for tt in natsort.natsorted(masks,reverse=False):
        #print(os.path.join(p,tt))
        maks = np.load(os.path.join(p+"\\"+"masks",tt))
        mask1 = resize(maks, (512, 512), mode='constant', preserve_range=True)
        ttt=tt.split('.')[0]
        #filename = "%d.png"%d
        #cv2.imwrite(trainmasks+'\\'+filename,mask1*(255/8)) # save images
        cv2.imwrite(os.path.join(trainmasks,str(i)+'_'+str(ttt.split('_')[0]+'_image'+'_label.png')),mask1)
        d+=1
    #print(i)
#ttt=tt.split('.')[0]
#os.path.join(trainmasks,str(i)+'_'+str(ttt.split('_')[0]+'_image'+'_label.png'))  
#%% 2017aapm testing images
import natsort
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
from tqdm import tqdm_notebook, tnrange
import sys
import os
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
#print(os.listdir(path))

def truncated_range(img):
    max_hu = 384
    min_hu = -384
    img[np.where(img > max_hu)] = max_hu
    img[np.where(img < min_hu)] = min_hu
    return (img - min_hu) / (max_hu - min_hu) * 255. 
path1='D:\\Newdatasetandcodes\\OARnewdataset\\Testoutput111'
directory_list=sorted(os.listdir(path1))
saveslicesimage='D:\\Newdatasetandcodes\\OARnewdataset\\testimagesfolder11'
#trainmasks='D:\\Newdatasetandcodes\\OARnewdataset\\trainmasks'
d=1
for i in natsort.natsorted(directory_list,reverse=False):
    p=os.path.join(path1,i)
    #print(p)
    imageid = next(os.walk(p+"\\"+"images"))
    images=imageid[2]
    #print(images)
#    maskid = next(os.walk(p+"\\"+"masks"))
#    masks=maskid[2]
    #ii.split('.')[0]
    os.mkdir(os.path.join(saveslicesimage,i))
    for ii in natsort.natsorted(images,reverse=False):
        print(os.path.join(p,ii))
        img = np.load(os.path.join(p+"\\"+"images",ii))
        stacked_img = np.stack((img,)*3, axis=-1)
        #imagef = img[:,:]
        #imagef= resize(imagef, (512, 512, 3), mode='constant', preserve_range=True)
        #img=truncated_range(imagef)
        #filename = "%d.png"%d
        cv2.imwrite(os.path.join(saveslicesimage+'\\'+str(i),str(i)+'_'+str(ii.split('.')[0]+'.png')),stacked_img) # save images
        d+=1 
    print(i)
#%%
import natsort
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
from tqdm import tqdm_notebook, tnrange
import sys
import os
#print(os.listdir(path))

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
#from keras import metrics
#from keras.utils import np_utils, plot_model
#from keras.models import Model, load_model
#from keras.layers import Input, concatenate, Dropout
#from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
#from keras.layers.pooling import MaxPooling2D
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras import backend as K
#import tensorflow as tf
#from keras.layers.core import Lambda, Dense, Dropout, Activation, Flatten, Reshape, Permute
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
path1='D:\\Newdatasetandcodes\\OARnewdataset\\Testoutput111'
directory_list=sorted(os.listdir(path1))
trainmasks1='D:\\Newdatasetandcodes\\OARnewdataset\\testimagemask11'
d=1
for i in natsort.natsorted(directory_list,reverse=False):
    p=os.path.join(path1,i)
    #print(p)
    maskid = next(os.walk(p+"\\"+"masks"))
    masks=maskid[2]
    os.mkdir(os.path.join(trainmasks1,i))
    for tt in natsort.natsorted(masks,reverse=False):
        #print(os.path.join(p,tt))
        maks = np.load(os.path.join(p+"\\"+"masks",tt))
        mask1 = resize(maks, (512, 512), mode='constant', preserve_range=True)
        ttt=tt.split('.')[0]
        #filename = "%d.png"%d
        #cv2.imwrite(trainmasks+'\\'+filename,mask1*(255/8)) # save images
        cv2.imwrite(os.path.join(trainmasks1+'\\'+str(i),str(i)+'_'+str(ttt.split('_')[0]+'_image'+'_label.png')),mask1)
        d+=1
    
#%% convert 2017 aapm dataset from dicom to nifty format
import os
import dicom2nifti
import os
import numpy as np
import matplotlib.pyplot as mpplot
import matplotlib.image as mpimg
import pydicom
import glob
import cv2
images = []
folders = glob.glob('D:\\Newdatasetandcodes\\OARnewdataset\\moditestimages\\*\\*')
path='D:\\Newdatasetandcodes\\OARnewdataset\\moditestimages\\'
patient_pattern = os.path.join(path,'*','*','*')
patient_list = glob.glob(patient_pattern)
output_folder='D:\\Newdatasetandcodes\\OARnewdataset\\testniftimages'
d=1
for patients in patient_list:
    print(os.path.join(path,patients))
    #dd=patients.split('\\')[-1]
    dd1=patients.split('\\')[4]
    cur_path=os.path.join(output_folder,dd1)
    os.mkdir(cur_path)
    dicom2nifti.convert_directory(os.path.join(path,patients), cur_path, compression=True, reorient=True)  
    
#%%
import nibabel as nib
import cv2
import numpy as np
import os
import natsort
#result_array = np.empty((512, 512,3))
#
#for line in data_array:
#    result = do_stuff(line)
#    result_array = np.append(result_array, [result], axis=0)
path1='D:\\Newdatasetandcodes\\OARnewdataset\\testimagemask1'
oslist=os.listdir(path1)
filesnew=natsort.natsorted(oslist)
#train_ids = next(os.walk(path1+"Patient"))[1]
#for ii in oslist:
#    print(os.path.join(data_pathnew,ii))
#    list2=os.path.join(data_pathnew,ii)
#    for jj in list2:
#        print(jj)
#        file=os.path.join(data_pathnew,ii)
#        print(os.path.join(data_pathnew,ii))
import natsort
import matplotlib.pyplot as plt
im_height=512
im_width=512
from tqdm import tqdm
for i, volume in enumerate(filesnew):
    #print(i)
    print(volume)
    cur_path = os.path.join(path1, volume)
    files=natsort.natsorted(os.listdir(cur_path))
    #train_ids = next(os.walk(path1+"images"))
    X_train = np.zeros((len(files),im_height, im_width,), dtype=np.uint8)
    stacked = np.zeros((im_height, im_width,len(files)),dtype=np.uint8)
    for n, id_ in tqdm(enumerate(files), total=len(files)):
        print(n)
        print(id_)
        img=cv2.imread(os.path.join(cur_path,id_))
        print(img.shape)
        x = np.array(img)[:,:,1]
        X_train[n] = x
        stacked[:,:,n]=x
        ff=np.swapaxes(X_train,0,2)
    i=i
    ex="D:\\Newdatasetandcodes\\OARnewdataset\\testinputniiimages/Patient_"+str(i)+".nii.gz"    
    img = nib.load(ex)
    im1=np.array(img.get_affine())
    new_image = nib.Nifti1Image(np.asarray(stacked,dtype="uint8" ), affine=im1)
    nib.save(new_image,'D:\\Newdatasetandcodes\\OARnewdataset\\testmasknii4/Patient_'+str(i)+'_GT.nii.gz')
    
    
#%%
import nibabel as nib
import cv2
import numpy as np
import os
import natsort
#result_array = np.empty((512, 512,3))
#
#for line in data_array:
#    result = do_stuff(line)
#    result_array = np.append(result_array, [result], axis=0)
path1='D:\\Newdatasetandcodes\\OARnewdataset\\testimagesfolder11'
oslist=os.listdir(path1)
filesnew=natsort.natsorted(oslist)
#train_ids = next(os.walk(path1+"Patient"))[1]
#for ii in oslist:
#    print(os.path.join(data_pathnew,ii))
#    list2=os.path.join(data_pathnew,ii)
#    for jj in list2:
#        print(jj)
#        file=os.path.join(data_pathnew,ii)
#        print(os.path.join(data_pathnew,ii))

import matplotlib.pyplot as plt
im_height=512
im_width=512
from tqdm import tqdm
for i, volume in enumerate(filesnew):
    #print(i)
    print(volume)
    cur_path = os.path.join(path1, volume)
    files=natsort.natsorted(os.listdir(cur_path))
    #train_ids = next(os.walk(path1+"images"))
    temp = []
    X_train = np.zeros((len(files),im_height, im_width), dtype=np.uint8)
    stacked = np.zeros((len(files),im_height, im_width),dtype=np.uint8)
    for n, id_ in tqdm(enumerate(files), total=len(files)):
        print(n)
        print(id_)
        img=cv2.imread(os.path.join(cur_path,id_))
        print(img.shape)
        x = np.array(img)[:,:,1]
        X_train[n] = x
        temp.append(img[:,:,1])
        dd=np.stack(X_train, axis=0)
        ff=np.swapaxes(X_train,2,0)
#        tt1=dd.transpose(2,1,0)
#        gg=dd.transpose(2,0,1)
#        #uu=dd.transpose(2,1,0)
#        ggg=gg.transpose(0,2,1)
#        stacked[n,:,:]=x
#        stacked1=np.swapaxes(stacked,0,2)
        #ttf=stacked1.transpose(1,2,0)
    i=i
    ex="D:\\Newdatasetandcodes\\OARnewdataset\\testinputniiimages/Patient_"+str(i)+".nii.gz"    
    img = nib.load(ex)
    im1=np.array(img.get_affine())
    new_image = nib.Nifti1Image(np.asarray(ff,dtype="uint8" ), affine=im1)
    nib.save(new_image,'D:\\Newdatasetandcodes\\OARnewdataset\\testimagenii1new1/Patient_'+str(i)+'_Image.nii.gz')
#temp1=temp[0]
#temp11=temp1[:,:,1]
#ff=np.swapaxes(dd,2,0)
#gg=dd.transpose(2,0,1)
##uu=dd.transpose(2,1,0)
#ggg=gg.transpose(0,2,1)
#from time import time
#import numpy as np
#stacked = np.empty(im_height, im_width,len(files))
## Create 100 images of the same dimention 256x512 (8-bit). 
## In reality, each image comes from a different file
#img = np.zeros((512, 512, 100))
#stacked = np.empty((im_height, im_width,len(files)),dtype=np.uint8)
#for n in range(len(files)):
#    stacked[:,:,n] = temp[n]
#ggn=new_image.get_data()
#ttf=stacked1.transpose(1,2,0)
#    
#%% For test dataset ######################################## AAPM 2019 dataset
import os, sys, glob

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

ROI_ORDER = ['SpinalCord', 'Lung_R', 'Lung_L', 'Heart', 'Esophagus']


def read_data(path, train_flag, floder_flag, out_path):
    '''
     read training set and validation set according to directory structure
    :param path:
    :param train_flag: future work for semi-supervised method to generate new masks
    :param floder_flag: output folder
    :param out_path: output path
    :return:
    '''
    for root, dirs, file in os.walk(path):
        for i in dirs:
            # get the number of current path
            dicoms = glob.glob(os.path.join(root + '\\' + i, '*.dcm'))

            # read RT structure files
            if train_flag == 'train_set' and 'simplified' in os.path.split(i)[1]:
                rt_dicom = pydicom.dcmread(dicoms[0])
                contours = get_contours(rt_dicom)

            # read dicoms
            elif len(dicoms) > 0:
                scans = [None] * len(dicoms)
                for i in range(len(dicoms)):
                    scans[i] = pydicom.dcmread(dicoms[i])
                    # print(scans[i].pixel_array.shape)
                    # plt.imshow(scans[i].pixel_array, cmap=plt.cm.bone)
                    # plt.show()

                # ['ImagePositionPatient', 'PatientPosition', 'PositionReferenceIndicator']
                print(scans[i].dir('position'))
                # print(scans[i].ImagePositionPatient)
                # sort by z
                scans.sort(key=lambda z: float(z.ImagePositionPatient[2]))
                #
                images = np.stack([s.pixel_array for s in scans])
                images = images.astype(np.int16)
                print(images.shape)

                # HU
                for s_num in range(len(scans)):
                    intercept = scans[s_num].RescaleIntercept
                    slope = scans[s_num].RescaleSlope
                    # print('inter:', intercept, 'slope', slope)

                    if slope != 1:
                        images[s_num] = slope * images[s_num].astype(np.float64)
                        images[s_num] = images[s_num].astype(np.int16)

                    images[s_num] += np.int16(intercept)
                    # plot HU for every image
                    # plt.hist(images[s_num].flatten(), bins=80, color='c')
                    # plt.show()

    # normalization HU
    images = normalization_hu(images)
    # plt.hist(images[s_num].flatten(), bins=80, color='c')
    # plt.show()

    if train_flag == 'train_set':

        # save images as npy
        for img in range(0, len(images)):
            total_imgs = []
            folder = os.path.exists(out_path + '/test_' + str(floder_flag) + '/images')
            if not folder:
                os.mkdir(out_path + '/test_' + str(floder_flag))
                os.mkdir(out_path + '/test_' + str(floder_flag) + '/images')

            if img == 0 or img == len(images) - 1:
                continue
            # print(img)
            # 2.5D data, using adjacent 3 images
            cur_img = images[img - 1:img + 2, :, :].astype('uint8')
            np.save(out_path + '/test_' + str(floder_flag) + '/images/' + str(img) + '_image.npy', cur_img)


        masks = get_masks(contours, images.shape, scans)

        # save masks as npy
        for msk in range(0, len(masks)):
            # save as png
            # labels_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
            # print('base name', os.path.basename(root))
            folder = os.path.exists(out_path + '/test_' + str(floder_flag) + "/masks")
            if not folder:
                os.mkdir(out_path + '/test_' + str(floder_flag) + "/masks")
            if msk == 0 or msk == len(images) - 1:
                continue
            # cur_mask = masks[msk - 1: msk + 2, :, :].astype('uint8')
            # cur_mask = masks[msk, :, :].astype('uint8')
            np.save(out_path + '/test_' + str(floder_flag) + '/masks/' + str(msk) + '_mask.npy', masks[msk])

        return images, masks

    # future work for semi-supervised learning
    # elif train_flag == 'semi_supervised_set':
    #     # save images as png
    #     # need to be un comment
    #     for img in range(0, len(images)):
    #         # images_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
    #         folder = os.path.exists(out_path + '/test_' + str(floder_flag) + "/images")
    #         if not folder:
    #             os.mkdir(out_path + "/images")
    #         else:
    #             np.save(out_path + '/train_' + str(floder_flag) + '/images/' + str(img) + '_image.npy', cur_img)
    #     return images


# some code imported from http://aapmchallenges.cloudapp.net/forums/3/2/
def get_contours(structure):
    '''
    get structure and reference information from RT Structure DICOM file
    :param structure: RT Structure file
    :return: contour
    '''
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
    return contours


def get_masks(contours, shape, scans):
    '''
    convert RT Structure to label map mask
    :param contours
    :param shape:
    :param scans:
    :return:
    '''
    z = [np.around(s.ImagePositionPatient[2], 1) for s in scans]
    pos_row = scans[0].ImagePositionPatient[1]
    # print(scans[0].ImagePositionPatient)
    spacing_row = scans[0].PixelSpacing[1]
    pos_column = scans[0].ImagePositionPatient[0]
    spacing_column = scans[0].PixelSpacing[0]

    mask = np.zeros(shape, dtype=np.float32)
    for con in contours:
        num = ROI_ORDER.index(con['name']) + 1
        for c in con['contours']:
            nodes = np.array(c).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = z.index(np.around(nodes[0, 2], 1))
            # print('z_index: ', z_index)
            r = (nodes[:, 1] - pos_row) / spacing_row
            c = (nodes[:, 0] - pos_column) / spacing_column
            rr, cc = polygon(r, c)
            mask[z_index, rr, cc] = int(num)
    # print('mask shape: ', mask[0], 'num shape', num)
    return mask
# import code end


def normalization_hu(images):
    '''
    normalization to 0-255. and CT value from -1000 to 800 HU
    :param images:
    :return: normalized image
    '''
    MIN = -1000
    MAX = 800
    images = (images - MIN) / (MAX - MIN)
    images[images > 1] = 1.
    images[images < 0] = 0.
    return images * 255


def plot_ct_scan(scan):
    '''
    plot a few more images of the slices
    :param scan:
    :return:
    '''
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(30, 30))  # set 7 for all images
    # print('shape[0]', scan.shape[0])
    for i in range(0, scan.shape[0], 5):
        for j in range(4):
            # plots[int(i / 20), j].axis('off')
            plots[int(i / 20), j].imshow(scan[i], cmap=plt.cm.bone)
    plt.show()


if __name__ == '__main__':
    # set the path
    total_data_path = ""
    total_data_path_val = "D:\\Newdatasetandcodes\\OARnewdataset\\testmodify\\Test"

    images_output_path = ""
    images_output_path_val = "D:\\Newdatasetandcodes\\OARnewdataset\\testoutput"

    cases = [os.path.join(total_data_path_val, name)
             for name in sorted(os.listdir(total_data_path_val)) if
             os.path.isdir(os.path.join(total_data_path_val, name))]
    print('Patient number: ', len(cases))

    # # train sets
    for c in cases:
        # for folder_flag in range(3):
        print('c index:', cases.index(c))
        folder_flag = cases.index(c)
        print('C: ', os.path.basename(os.path.dirname(c)))
        images, masks = read_data(c, 'train_set', folder_flag, images_output_path_val)

    plot_ct_scan(images)    
    
    
    
#%% check numpy array ########################AAPM2019 dataset ########################################
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
from tqdm import tqdm_notebook, tnrange
import sys
import os
#print(os.listdir(path))

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from keras import metrics
from keras.utils import np_utils, plot_model
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras.layers.core import Lambda, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

path="D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\AAPM2019out\\"
print(os.listdir(path))
plt.figure(figsize=(30,30))
for j in range(2):
    q = j+1
    j = j+1
    img = np.load(path+'\\'+'images\\' + str(30) + '_image.npy').transpose(1, 2, 0)
    img_mask = np.load(path+'\\'+'masks\\' + str(30) +'_mask.npy')
    
    plt.subplot(1,2*3,q*2-1)
    plt.imshow(img)
    plt.subplot(1,2*3,q*2)
    plt.imshow(img_mask)
plt.show()
path="D:\\Newdatasetandcodes\\OARnewdataset\\output\\train_0"
#ff=os.walk(path+"images")
#for i in ff:
#    print(i)
    
import os 
#for (root,dirs,files) in os.walk(path):
#    print(root)
#
#import os
#for root, dirs, files in os.walk(path, topdown=False):
#   for name in files:
#      print(os.path.join(root, name))
#   for name in dirs:
#      print(os.path.join(root, name))
#path='D:\\Newdatasetandcodes\\OARnewdataset\\output'
def train_idx(path):
    import re
    train_ids = next(os.walk(path+"\\"+"images"))
    train_ids=train_ids[2]
    print(len(train_ids))
    new_train_ids=[]
    for item in train_ids:
        new_train_ids.append(' '.join(re.findall(r'\d+', item)))
    print(new_train_ids)
    return new_train_ids
#new_train_ids=train_idx(path)
#Set the input shape
input_shape=(512, 512, 3)
im_width = input_shape[0]
im_height = input_shape[1]
n_labels = 6
def numpy_funct(path,new_train_ids):
    sys.stdout.flush()
    for n, id_ in tqdm_notebook(enumerate(new_train_ids), total=len(new_train_ids)):
        img = np.load(path +'\\'+ 'images\\' + id_+'_image.npy').transpose(1, 2, 0)
        x = img[:,:,:]
        x = resize(x, (512, 512, 3), mode='constant', preserve_range=True)
        X_train[n] = x
    
        mask = (np.load(path +'\\'+ 'masks\\' + id_+'_mask.npy'))
        Y_train[n] = resize(mask, (512, 512), mode='constant', preserve_range=True)
    return  X_train,Y_train

#% Convert into numpy array#######################################################3

#path1='D:\\Newdatasetandcodes\\OARnewdataset\\output'
#directory_list=sorted(os.listdir(path1))
#import natsort
#input_shape=(512, 512, 3)
#im_width = input_shape[0]
#im_height = input_shape[1]
#n_labels = 6
##X_train = np.zeros((len(new_train_ids), im_height, im_width, 3), dtype=np.uint8)
##Y_train = np.zeros((len(new_train_ids), im_height, im_width), dtype=np.uint8)
#Train_Xoverll=[]
#Train_Yoverall=[]
#for i in natsort.natsorted(directory_list,reverse=False):
#    p=os.path.join(path1,i)
#    print(p)
#    new_train_ids=train_idx(p)
#    X_train = np.zeros((len(new_train_ids), im_height, im_width, 3), dtype=np.uint8)
#    Y_train = np.zeros((len(new_train_ids), im_height, im_width), dtype=np.uint8)
#    X_train,Y_train=numpy_funct(p,new_train_ids)
#    Train_Xoverll.append(X_train)
#    Train_Yoverall.append(Y_train)
#    print(Y_train.shape)
##np_utils.to_categorical for one hot encoding to no of label 
#Y_train = np_utils.to_categorical(Y_train, n_labels)
#print(X_train.shape)
#print(Y_train.shape) 
#from PIL import Image
#gh=np.array(Train_Yoverall)
#gh.shape
#dfg=X_train

#%%                       conver numpy into images 2017 AAMP dataset
import natsort
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
from tqdm import tqdm_notebook, tnrange
import sys
import os
#print(os.listdir(path))

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from keras import metrics
from keras.utils import np_utils, plot_model
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras.layers.core import Lambda, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
path1='D:\\Newdatasetandcodes\\OARnewdataset\\output'
directory_list=sorted(os.listdir(path1))
saveslicesimage='D:\\Newdatasetandcodes\\OARnewdataset\\Trainimages'
trainmasks='D:\\Newdatasetandcodes\\OARnewdataset\\trainmasks'
d=1
for i in natsort.natsorted(directory_list,reverse=False):
    p=os.path.join(path1,i)
    #print(p)
    imageid = next(os.walk(p+"\\"+"images"))
    images=imageid[2]
#    maskid = next(os.walk(p+"\\"+"masks"))
#    masks=maskid[2]
    for ii in natsort.natsorted(images,reverse=False):
        #print(os.path.join(p,ii))
        img = np.load(os.path.join(p+"\\"+"images",ii)).transpose(1, 2, 0)
        imagef = img[:,:,:]
        imagef= resize(imagef, (512, 512, 3), mode='constant', preserve_range=True)
        filename = "%d.png"%d
        cv2.imwrite(saveslicesimage+'\\'+filename,imagef) # save images
        d+=1 
    print(i)
#    for tt in natsort.natsorted(masks,reverse=False):
#        #print(os.path.join(p,ii))
#        maks = np.load(os.path.join(p+"\\"+"masks",tt))
#        mask1 = resize(maks, (512, 512), mode='constant', preserve_range=True)
#        filename = "%d.png"%d
#        cv2.imwrite(trainmasks+'\\'+filename,mask1*(255/5)) # save images
#        d+=1
        #cv2.imwrite('color_img.jpg', x)
        #print(x.mask1)
#%% AAPM 2019 dataset 
import natsort
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
from tqdm import tqdm_notebook, tnrange
import sys
import os
#print(os.listdir(path))

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from keras import metrics
from keras.utils import np_utils, plot_model
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras.layers.core import Lambda, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
def truncated_range(img):
    max_hu = 384
    min_hu = -384
    img[np.where(img > max_hu)] = max_hu
    img[np.where(img < min_hu)] = min_hu
    return (img - min_hu) / (max_hu - min_hu) * 255. 
path1='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\Trainingdata'
directory_list=sorted(os.listdir(path1))
saveslicesimage='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\Trainingimages'
#trainmasks='D:\\Newdatasetandcodes\\OARnewdataset\\trainmasks'
d=1
for i in natsort.natsorted(directory_list,reverse=False):
    p=os.path.join(path1,i)
    #print(p)
    imageid = next(os.walk(p+"\\"+"images"))
    images=imageid[2]
    #print(images)
#    maskid = next(os.walk(p+"\\"+"masks"))
#    masks=maskid[2]
    #ii.split('.')[0]
    for ii in natsort.natsorted(images,reverse=False):
        print(os.path.join(p,ii))
        img = np.load(os.path.join(p+"\\"+"images",ii)).transpose(1, 2, 0)
        imagef = img[:,:,:]
        imagef= resize(imagef, (512, 512, 3), mode='constant', preserve_range=True)
        #img=truncated_range(imagef)
        filename = "%d.png"%d
        cv2.imwrite(os.path.join(saveslicesimage,str(i)+'_'+str(ii.split('.')[0]+'.png')),imagef) # save images
        d+=1 
    print(i)
#    for tt in natsort.natsorted(masks,reverse=False):
#        #print(os.path.join(p,ii))
#        maks = np.load(os.path.join(p+"\\"+"masks",tt))
#        mask1 = resize(maks, (512, 512), mode='constant', preserve_range=True)
#        filename = "%d.png"%d
#        cv2.imwrite(trainmasks+'\\'+filename,mask1*(255/5)) # save images
#        d+=1
        #cv2.imwrite('color_img.jpg', x)
        #print(x.mask1) 
        
#os.path.join(saveslicesimage,str(i)+'_'+str(ii.split('.')[0]+'.png'))        
#        
#import numpy as np      
#  
#cv2.imwrite('D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\testing1\\Patient_'+str(i)+'/Patient_'+str(i)+'_'+str(l)+'_image_'+'_label.png', lbl_array)
#
#'_image_label.png'
#tt=truncated_range(data001)
#ttt=tt[:,:,1]
#def normalization_hu(images):
#    '''
#    normalization to 0-255. and CT value from -1000 to 800 HU
#    :param images:
#    :return: normalized image
#    '''
#    MIN = -1000
#    MAX = 800
#    images = (images - MIN) / (MAX - MIN)
#    images[images > 1] = 1.
#    images[images < 0] = 0.
#    return images * 255
#ff=normalization_hu(data)
#data1=data000[:,:,2]
#ff1=ff[:,:,1]   
#%%                          convert into mask 2019 AAPM dataset
import natsort
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
from tqdm import tqdm_notebook, tnrange
import sys
import os
#print(os.listdir(path))

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from keras import metrics
from keras.utils import np_utils, plot_model
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras.layers.core import Lambda, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
path1='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\Trainingdata'
directory_list=sorted(os.listdir(path1))
trainmasks='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\Trainingmask'
d=1
for i in natsort.natsorted(directory_list,reverse=False):
    p=os.path.join(path1,i)
    #print(p)
    maskid = next(os.walk(p+"\\"+"masks"))
    masks=maskid[2]
    for tt in natsort.natsorted(masks,reverse=False):
        #print(os.path.join(p,tt))
        maks = np.load(os.path.join(p+"\\"+"masks",tt))
        mask1 = resize(maks, (512, 512), mode='constant', preserve_range=True)
        ttt=tt.split('.')[0]
        #filename = "%d.png"%d
        #cv2.imwrite(trainmasks+'\\'+filename,mask1*(255/8)) # save images
        cv2.imwrite(os.path.join(trainmasks,str(i)+'_'+str(ttt.split('_')[0]+'_image'+'_label.png')),mask1)
        d+=1
    #print(i)
#ttt=tt.split('.')[0]
#os.path.join(trainmasks,str(i)+'_'+str(ttt.split('_')[0]+'_image'+'_label.png'))  
#%% ###################################Testing AAPM 2019 dataset ##########################################
import natsort
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
from tqdm import tqdm_notebook, tnrange
import sys
import os
#print(os.listdir(path))

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from keras import metrics
from keras.utils import np_utils, plot_model
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras.layers.core import Lambda, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
def truncated_range(img):
    max_hu = 384
    min_hu = -384
    img[np.where(img > max_hu)] = max_hu
    img[np.where(img < min_hu)] = min_hu
    return (img - min_hu) / (max_hu - min_hu) * 255. 
path1='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\Testingdata'
directory_list=sorted(os.listdir(path1))
saveslicesimage='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\Testimages'
#trainmasks='D:\\Newdatasetandcodes\\OARnewdataset\\trainmasks'
d=1
for i in natsort.natsorted(directory_list,reverse=False):
    p=os.path.join(path1,i)
    #print(p)
    imageid = next(os.walk(p+"\\"+"images"))
    images=imageid[2]
    #print(images)
#    maskid = next(os.walk(p+"\\"+"masks"))
#    masks=maskid[2]
    #ii.split('.')[0]
    for ii in natsort.natsorted(images,reverse=False):
        print(os.path.join(p,ii))
        img = np.load(os.path.join(p+"\\"+"images",ii)).transpose(1, 2, 0)
        imagef = img[:,:,:]
        imagef= resize(imagef, (512, 512, 3), mode='constant', preserve_range=True)
        #img=truncated_range(imagef)
        filename = "%d.png"%d
        cv2.imwrite(os.path.join(saveslicesimage,str(i)+'_'+str(ii.split('.')[0]+'.png')),imagef) # save images
        d+=1 
    print(i)
#    for tt in natsort.natsorted(masks,reverse=False):
#        #print(os.path.join(p,ii))
#        maks = np.load(os.path.join(p+"\\"+"masks",tt))
#        mask1 = resize(maks, (512, 512), mode='constant', preserve_range=True)
#        filename = "%d.png"%d
#        cv2.imwrite(trainmasks+'\\'+filename,mask1*(255/5)) # save images
#        d+=1
        #cv2.imwrite('color_img.jpg', x)
        #print(x.mask1) 
        
#os.path.join(saveslicesimage,str(i)+'_'+str(ii.split('.')[0]+'.png'))        
#        
#import numpy as np      
#  
#cv2.imwrite('D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\testing1\\Patient_'+str(i)+'/Patient_'+str(i)+'_'+str(l)+'_image_'+'_label.png', lbl_array)
#
#'_image_label.png'
#tt=truncated_range(data001)
#ttt=tt[:,:,1]
#def normalization_hu(images):
#    '''
#    normalization to 0-255. and CT value from -1000 to 800 HU
#    :param images:
#    :return: normalized image
#    '''
#    MIN = -1000
#    MAX = 800
#    images = (images - MIN) / (MAX - MIN)
#    images[images > 1] = 1.
#    images[images < 0] = 0.
#    return images * 255
#ff=normalization_hu(data)
#data1=data000[:,:,2]
#ff1=ff[:,:,1]     
#%%
import natsort
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
from tqdm import tqdm_notebook, tnrange
import sys
import os
#print(os.listdir(path))

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from keras import metrics
from keras.utils import np_utils, plot_model
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras.layers.core import Lambda, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
path1='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\Testingdata'
directory_list=sorted(os.listdir(path1))
trainmasks='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\testmask'
d=1
for i in natsort.natsorted(directory_list,reverse=False):
    p=os.path.join(path1,i)
    #print(p)
    maskid = next(os.walk(p+"\\"+"masks"))
    masks=maskid[2]
    for tt in natsort.natsorted(masks,reverse=False):
        #print(os.path.join(p,tt))
        maks = np.load(os.path.join(p+"\\"+"masks",tt))
        mask1 = resize(maks, (512, 512), mode='constant', preserve_range=True)
        ttt=tt.split('.')[0]
        #filename = "%d.png"%d
        #cv2.imwrite(trainmasks+'\\'+filename,mask1*(255/8)) # save images
        cv2.imwrite(os.path.join(trainmasks,str(i)+'_'+str(ttt.split('_')[0]+'_image'+'_label.png')),mask1)
        d+=1
    
#%% Testing png images and masks ############################################AAPM 2017 #######################
import natsort
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
from tqdm import tqdm_notebook, tnrange
import sys
import os
#print(os.listdir(path))

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from keras import metrics
from keras.utils import np_utils, plot_model
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras.layers.core import Lambda, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
path1='D:\\Newdatasetandcodes\\OARnewdataset\\testoutput'
directory_list=sorted(os.listdir(path1))
saveslicesimage='D:\\Newdatasetandcodes\\OARnewdataset\\Testimages'
d=1
for i in natsort.natsorted(directory_list,reverse=False):
    p=os.path.join(path1,i)
    #print(p)
    imageid = next(os.walk(p+"\\"+"images"))
    images=imageid[2]
#    maskid = next(os.walk(p+"\\"+"masks"))
#    masks=maskid[2]
    for ii in natsort.natsorted(images,reverse=False):
        #print(os.path.join(p,ii))
        img = np.load(os.path.join(p+"\\"+"images",ii)).transpose(1, 2, 0)
        imagef = img[:,:,:]
        imagef= resize(imagef, (512, 512, 3), mode='constant', preserve_range=True)
        filename = "%d.png"%d
        cv2.imwrite(saveslicesimage+'\\'+filename,imagef) # save images
        d+=1 
    print(i)
#%% Testing png masks###################################### AAMP2017########################3
import natsort
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2
from tqdm import tqdm_notebook, tnrange
import sys
import os
#print(os.listdir(path))

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from keras import metrics
from keras.utils import np_utils, plot_model
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras.layers.core import Lambda, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
path1='D:\\Newdatasetandcodes\\OARnewdataset\\testoutput'
directory_list=sorted(os.listdir(path1))
trainmasks='D:\\Newdatasetandcodes\\OARnewdataset\\Testing mask'
d=1
for i in natsort.natsorted(directory_list,reverse=False):
    p=os.path.join(path1,i)
    #print(p)
    maskid = next(os.walk(p+"\\"+"masks"))
    masks=maskid[2]
    for tt in natsort.natsorted(masks,reverse=False):
        #print(os.path.join(p,ii))
        maks = np.load(os.path.join(p+"\\"+"masks",tt))
        mask1 = resize(maks, (512, 512), mode='constant', preserve_range=True)
        filename = "%d.png"%d
        cv2.imwrite(trainmasks+'\\'+filename,mask1*(255/5)) # save images
        d+=1
    print(i)
#%% ##################################### Converting png into numpy array ###################### 2017 AAPM
#path_train='C:\\Users\\abdul\\Desktop\\kagglesalt\\tgs-salt-identification-challenge\\train\\'

path_train='D:\\Newdatasetandcodes\\OARnewdataset\\'
import os
import numpy as np
train_ids = next(os.walk(path_train+"Trainimages"))[2]
#test_ids = next(os.walk(path_test+"images"))[2]

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

im_height=512
im_width=512
im_chan=3
X_train = np.zeros((len(train_ids), im_height, im_width, 3), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), im_height, im_width), dtype=np.uint8)
print('Getting and resizing train images and masks ... ')
#sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = path_train
    #print(n)
    #print(id_)
    img = imread(path + '/Trainimages/' + id_)
    x = np.array(img)
    x = resize(x, (512, 512, 3), mode='constant', preserve_range=True)
    X_train[n] = x
    p=imread(path + '/trainmasks/' + id_)
    #mask = np.expand_dims((p)[:,:,1],
    mask=resize(p, (512, 512), mode='constant', preserve_range=True)                    
    Y_train[n] =mask 

print('Done!')
ix = random.randint(0, len(train_ids))
plt.imshow(np.dstack((X_train[ix],X_train[ix],X_train[ix])))
plt.show()
tmp = np.squeeze(Y_train[ix]).astype(np.float32)
plt.imshow(np.dstack((tmp,tmp,tmp)))
plt.show()
import numpy as np
np.save('Training_images1.npy',X_train)
np.save('Training_masks1.npy',Y_train)
loadtraining=np.load('Training_masks1.npy')
n_labels=6
from keras.utils import np_utils, plot_model
Y_train = np_utils.to_categorical(loadtraining, n_labels)
#def numpy_funct(path,new_train_ids):
#    sys.stdout.flush()
#    for n, id_ in tqdm_notebook(enumerate(new_train_ids), total=len(new_train_ids)):
#        img = np.load(path +'\\'+ 'images\\' + id_+'_image.npy').transpose(1, 2, 0)
#        x = img[:,:,:]
#        x = resize(x, (512, 512, 3), mode='constant', preserve_range=True)
#        X_train[n] = x
#    
#        mask = (np.load(path +'\\'+ 'masks\\' + id_+'_mask.npy'))
#        Y_train[n] = resize(mask, (512, 512), mode='constant', preserve_range=True)
#    return  X_train,Y_train
#
##% Convert into numpy array#######################################################3
#
#path1='D:\\Newdatasetandcodes\\OARnewdataset\\output'
#directory_list=sorted(os.listdir(path1))
#import natsort
#input_shape=(512, 512, 3)
#im_width = input_shape[0]
#im_height = input_shape[1]
#n_labels = 6
##X_train = np.zeros((len(new_train_ids), im_height, im_width, 3), dtype=np.uint8)
##Y_train = np.zeros((len(new_train_ids), im_height, im_width), dtype=np.uint8)
#Train_Xoverll=[]
#Train_Yoverall=[]
#for i in natsort.natsorted(directory_list,reverse=False):
#    p=os.path.join(path1,i)
#    print(p)
#    new_train_ids=train_idx(p)
#    X_train = np.zeros((len(new_train_ids), im_height, im_width, 3), dtype=np.uint8)
#    Y_train = np.zeros((len(new_train_ids), im_height, im_width), dtype=np.uint8)

# Check if training data looks all right
ix = random.randint(0, len(train_ids))
plt.imshow(np.dstack((X_train[ix],X_train[ix],X_train[ix])))
plt.show()
tmp = np.squeeze(Y_train[ix]).astype(np.float32)
plt.imshow(np.dstack((tmp,tmp,tmp)))
plt.show()
#%%#############################3 new function for numpy array############################### 2-17 AAPM
path_train='D:\\Newdatasetandcodes\\OARnewdataset\\'
import os
import numpy as np
train_ids = next(os.walk(path_train+"Testimages"))[2]
#test_ids = next(os.walk(path_test+"images"))[2]

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

im_height=512
im_width=512
im_chan=3
X_train = np.zeros((len(train_ids), im_height, im_width, 3), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), im_height, im_width), dtype=np.uint8)
print('Getting and resizing train images and masks ... ')
#sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = path_train
    #print(n)
    #print(id_)
    img = imread(path + '/Testimages/' + id_)
    x = np.array(img)
    x = resize(x, (512, 512, 3), mode='constant', preserve_range=True)
    X_train[n] = x
    p=imread(path + '/Testing mask/' + id_)
    #mask = np.expand_dims((p)[:,:,1],
    mask=resize(p, (512, 512), mode='constant', preserve_range=True)                    
    Y_train[n] =mask 

import numpy as np
np.save('Testing_images.npy',X_train)
np.save('Testing_masks.npy',Y_train)
loadtraining=np.load('Training_images.npy')
#X_train.shape
#X_train1=X_train[1,:,:,:]
#X_train11=X_train1[:,:,2]
#plt.imshow(X_train11)
Y_train.shap
n_labels=6
from keras.utils import np_utils, plot_model
Y_train = np_utils.to_categorical(Y_train, n_labels)
print('Done!')
ix = random.randint(0, len(train_ids))
plt.imshow(np.dstack((X_train[ix],X_train[ix],X_train[ix])))
plt.show()
tmp = np.squeeze(Y_train[ix]).astype(np.float32)
plt.imshow(np.dstack((tmp,tmp,tmp)))
plt.show()

#%% https://github.com/wxkkk/OARs_Seg
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou    
#%
inputs = Input((input_shape))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(8, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (s)
c1 = Conv2D(8, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2),kernel_initializer='he_normal', padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2),kernel_initializer='he_normal', padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2),kernel_initializer='he_normal', padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2),kernel_initializer='he_normal', padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(n_labels, (1, 1), activation='softmax') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='categorical_crossentropy' , metrics=['categorical_accuracy', iou_coef])
model.summary()    

earlystopper = EarlyStopping(patience=10, verbose=1)
checkpointer = ModelCheckpoint('/content/drive/My Drive/Colab Notebooks/MSc_project/test/model.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.3, batch_size=32, epochs=100, 
                    callbacks=[earlystopper, checkpointer])

#model.save('/content/drive/My Drive/Colab Notebooks/MSc_project/test/model.h5')    
## Draw MIoU
#plt.plot(results.history['iou_coef'])
#plt.plot(results.history['val_iou_coef'])
#plt.title('Model iou_coef')
#plt.ylabel('iou_coef')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
#
## Draw Loss
#plt.plot(results.history['loss'])
#plt.plot(results.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
#%% testing the model
import numpy as np 
from keras.models import Model, load_model
from keras import backend as K
from skimage.transform import resize
from keras.utils import np_utils, plot_model
import matplotlib.pyplot as plt

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

# imported code from https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
        
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)
# import code end
    
dependencies = {
    'precision': precision,
    'recall': recall,
    'fmeasure': fmeasure,
    'iou_coef': iou_coef,
    'dice_coef': dice_coef
}

model = load_model('/content/drive/My Drive/Colab Notebooks/MSc_project/experiment/models/Model_B.h5', custom_objects=dependencies)

test_num = 24
X_test = np.zeros((test_num, im_height, im_width, 3), dtype=np.uint8)
Y_test = np.zeros((test_num, im_height, im_width), dtype=np.uint8)

for i in range(test_num):
    img = np.load(val_path + 'images/' + str(i+1) +'_image.npy').transpose(1, 2, 0)
    x = img[:,:,:]
    x = resize(x, (512, 512, 3), mode='constant', preserve_range=True)
    X_test[i] = x
    
    mask = (np.load(val_path + 'masks/' + str(i+1) +'_mask.npy'))
    Y_test[i] = resize(mask, (512, 512), mode='constant', preserve_range=True)

Y_test_one_hot = np_utils.to_categorical(Y_test, 6) #np_utils.to_categorical for one hot encoding to no of label 

print(model.evaluate(X_test, Y_test_one_hot))
print(model.metrics_names)

predicted_mask = model.predict(X_test)
test_out = predicted_mask[5,:,:,:]
test_decode = np.argmax(test_out, axis=-1)
plt.imshow(test_decode)

plt.figure(figsize=(20, 100))
for i in range(24):
  plt.subplot(24, 3, 3 * i + 1)
  plt.imshow(X_test[i])
  plt.title("Input image")
  
  plt.subplot(24, 3, 3 * i + 2)
  plt.imshow(Y_test[i])
  plt.title("Ground Truth Mask")
  plt.subplot(24, 3, 3 * i + 3)
  predicted = predicted_mask[i,:,:,:]
  predicted_decode = np.argmax(predicted, axis=-1)
  plt.imshow(predicted_decode)
  plt.title("Predicted Mask")
plt.suptitle("Predictions of model B on validation set")
plt.show()
#Visulization feature maps of any layer
# n_layer = 32
# get_layer_output = K.function([model.layers[0].input], [model.layers[n_layer].output])

# print(get_layer_output.name)
# layer_output = get_layer_output([X_test])[0]
# print(layer_output.shape)

# # plt.matshow(layer_output[1], cmap='viridis')
# plt.figure(figsize=(20, 10))

# for _ in range(128):
#             show_img = layer_output[0, :, :, _]
#             show_img.shape = [512, 512]
#             plt.subplot(4, 8, _ + 1)
#             plt.imshow(show_img, cmap='viridis')
#             plt.axis('off')
# plt.show()


    
    

#%%
imageid = next(os.walk(p+"\\"+"images"))
images=imageid[2]
maskid = next(os.walk(p+"\\"+"masks"))
masks=maskid[2]
for ii in natsort.natsorted(images,reverse=False):
    #print(os.path.join(p,ii))
    img = np.load(os.path.join(p+"\\"+"images",ii)).transpose(1, 2, 0)
    x = img[:,:,:]
    x = resize(x, (512, 512, 3), mode='constant', preserve_range=True)
    cv2.imwrite('color_img.jpg', x)
    print(im.shape)

for ii in natsort.natsorted(masks,reverse=False):
    #print(os.path.join(p,ii))
    maks = np.load(os.path.join(p+"\\"+"masks",ii))
    x = resize(maks, (512, 512), mode='constant', preserve_range=True)
    #cv2.imwrite('color_img.jpg', x)
    print(x.shape)

x1=img[:,:,2]
#path=p
#import numpy as np
for ii in os.listdir(p):
    images=os.path.join(p,ii)
    masks=os.path.join(p,ii)
    print(images)
def numpy_funct(path,new_train_ids):
    sys.stdout.flush()
    for n, id_ in tqdm_notebook(enumerate(new_train_ids), total=len(new_train_ids)):
        img = np.load(path +'\\'+ 'images\\' + id_+'_image.npy').transpose(1, 2, 0)
        x = img[:,:,:]
        x = resize(x, (512, 512, 3), mode='constant', preserve_range=True)
        X_train[n] = x
    
        mask = (np.load(path +'\\'+ 'masks\\' + id_+'_mask.npy'))
        Y_train[n] = resize(mask, (512, 512), mode='constant', preserve_range=True)
    return  X_train,Y_train
#Y_train,X_train=numpy_funct(path,new_train_ids)
#print(Y_train.shape)
##np_utils.to_categorical for one hot encoding to no of label 
#Y_train = np_utils.to_categorical(Y_train, n_labels)
#print(X_train.shape)
#print(Y_train.shape)  
ghh=np.array(gh)
gh1=gh[0]
gh2=gh[1]
gh3=gh[2]
array1=[]
for i in len(gh):
    array=gh[i]
    print(array)


merge_arr = np.concatenate([gh1, gh2, gh3], axis=0)
print(merge_arr.shape)  # (80, 4, 10, 10)

allArrays = np.concatenate([myFunction(x) for x in range])

#import tensorflow as tf
#
#im_height=128
#im_width=128
#im_chan=1
#X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
#Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool)
#print('Getting and resizing train images and masks ... ')
##sys.stdout.flush()
#for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
#    path = path_train
#    img = imread(path + '/images/' + id_)
#    x = np.array(img)[:,:,1]
#    x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
#    X_train[n] = x
#    p=imread(path + '/masks/' + id_)
#    #mask = np.expand_dims((p)[:,:,1],
#    mask=np.expand_dims(resize(p, (128, 128), mode='constant', preserve_range=True), axis=-1)                     
#    Y_train[n] =mask 
#
#print('Done!')
#
#
## Check if training data looks all right
#ix = random.randint(0, len(train_ids))
#plt.imshow(np.dstack((X_train[ix],X_train[ix],X_train[ix])))
#plt.show()
#tmp = np.squeeze(Y_train[ix]).astype(np.float32)
#plt.imshow(np.dstack((tmp,tmp,tmp)))
#plt.show()
  
#%% used this dataset for latest version AAPMRMTC ####################3
import os, sys, glob

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

#ROI_ORDER = ['Parotid_L','Parotid_R', 'Glnd_Submand_L', 'Glnd_Submand_R', 'LN_Neck_II_L', 'LN_Neck_II_R','LN_Neck_III_L','LN_Neck_III_R']

ROI_ORDER = ['Glnd_Submand_L', 'Glnd_Submand_R', 'LN_Neck_II_L', 'LN_Neck_II_R','LN_Neck_III_L','LN_Neck_III_R','Parotid_L','Parotid_R']

def read_data(path, train_flag, floder_flag, out_path):
    '''
     read training set and validation set according to directory structure
    :param path:
    :param train_flag: future work for semi-supervised method to generate new masks
    :param floder_flag: output folder
    :param out_path: output path
    :return:
    '''
    for root, dirs, file in os.walk(path):
        for i in dirs:
            # get the number of current path
            dicoms = glob.glob(os.path.join(root + '\\' + i, '*.dcm'))

            # read RT structure files
            if train_flag == 'train_set' and 'simplified' in os.path.split(i)[1]:
                rt_dicom = pydicom.dcmread(dicoms[0])
                contours = get_contours(rt_dicom)

            # read dicoms
            elif len(dicoms) > 0:
                scans = [None] * len(dicoms)
                for i in range(len(dicoms)):
                    scans[i] = pydicom.dcmread(dicoms[i])
                    # print(scans[i].pixel_array.shape)
                    # plt.imshow(scans[i].pixel_array, cmap=plt.cm.bone)
                    # plt.show()

                # ['ImagePositionPatient', 'PatientPosition', 'PositionReferenceIndicator']
                print(scans[i].dir('position'))
                # print(scans[i].ImagePositionPatient)
                # sort by z
                scans.sort(key=lambda z: float(z.ImagePositionPatient[2]))
                #
                images = np.stack([s.pixel_array for s in scans])
                images = images.astype(np.int16)
                print(images.shape)

                # HU
#                for s_num in range(len(scans)):
#                    intercept = scans[s_num].RescaleIntercept
#                    slope = scans[s_num].RescaleSlope
#                    # print('inter:', intercept, 'slope', slope)
#
#                    if slope != 1:
#                        images[s_num] = slope * images[s_num].astype(np.float64)
#                        images[s_num] = images[s_num].astype(np.int16)
#
#                    images[s_num] += np.int16(intercept)
                    # plot HU for every image
                    # plt.hist(images[s_num].flatten(), bins=80, color='c')
                    # plt.show()

    # normalization HU
    #images = normalization_hu(images)
    images=images
    # plt.hist(images[s_num].flatten(), bins=80, color='c')
    # plt.show()

    if train_flag == 'train_set':

        # save images as npy
        for img in range(0, len(images)):
            total_imgs = []
            folder = os.path.exists(out_path + '/train_' + str(floder_flag) + '/images')
            if not folder:
                os.mkdir(out_path + '/train_' + str(floder_flag))
                os.mkdir(out_path + '/train_' + str(floder_flag) + '/images')

            if img == 0 or img == len(images) - 1:
                continue
            # print(img)
            # 2.5D data, using adjacent 3 images
            cur_img = images[img - 1:img + 2, :, :].astype('uint8')
            np.save(out_path + '/train_' + str(floder_flag) + '/images/' + str(img) + '_image.npy', cur_img)


        masks = get_masks(contours, images.shape, scans)

        # save masks as npy
        for msk in range(0, len(masks)):
            # save as png
            # labels_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
            # print('base name', os.path.basename(root))
            folder = os.path.exists(out_path + '/train_' + str(floder_flag) + "/masks")
            if not folder:
                os.mkdir(out_path + '/train_' + str(floder_flag) + "/masks")
            if msk == 0 or msk == len(images) - 1:
                continue
            # cur_mask = masks[msk - 1: msk + 2, :, :].astype('uint8')
            # cur_mask = masks[msk, :, :].astype('uint8')
            np.save(out_path + '/train_' + str(floder_flag) + '/masks/' + str(msk) + '_mask.npy', masks[msk])

        return images, masks

    # future work for semi-supervised learning
    # elif train_flag == 'semi_supervised_set':
    #     # save images as png
    #     # need to be un comment
    #     for img in range(0, len(images)):
    #         # images_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
    #         folder = os.path.exists(out_path + '/test_' + str(floder_flag) + "/images")
    #         if not folder:
    #             os.mkdir(out_path + "/images")
    #         else:
    #             np.save(out_path + '/train_' + str(floder_flag) + '/images/' + str(img) + '_image.npy', cur_img)
    #     return images


# some code imported from http://aapmchallenges.cloudapp.net/forums/3/2/
def get_contours(structure):
    '''
    get structure and reference information from RT Structure DICOM file
    :param structure: RT Structure file
    :return: contour
    '''
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
    return contours


def get_masks(contours, shape, scans):
    '''
    convert RT Structure to label map mask
    :param contours
    :param shape:
    :param scans:
    :return:
    '''
    z = [np.around(s.ImagePositionPatient[2], 1) for s in scans]
    pos_row = scans[0].ImagePositionPatient[1]
    # print(scans[0].ImagePositionPatient)
    spacing_row = scans[0].PixelSpacing[1]
    pos_column = scans[0].ImagePositionPatient[0]
    spacing_column = scans[0].PixelSpacing[0]

    mask = np.zeros(shape, dtype=np.float32)
    for con in contours:
        num = ROI_ORDER.index(con['name']) + 1
        for c in con['contours']:
            nodes = np.array(c).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = z.index(np.around(nodes[0, 2], 1))
            # print('z_index: ', z_index)
            r = (nodes[:, 1] - pos_row) / spacing_row
            c = (nodes[:, 0] - pos_column) / spacing_column
            rr, cc = polygon(r, c)
            mask[z_index, rr, cc] = int(num)
    # print('mask shape: ', mask[0], 'num shape', num)
    return mask
# import code end
#path="D:\\Newdatasetandcodes\\AAPM2019\\testcase\\LCTSC-Train-S1-007\\12-13-2003-RTRCCTTHORAX8FHigh Adult-15875\\1-.simplified-17709"
##masks = get_masks(contours, images.shape, scans)
#for root, dirs, file in os.walk(path):
#    for i in file:
#        # get the number of current path
#        print(i)
#dicoms = glob.glob(os.path.join(root +'\\'+'*.dcm'))
#rt_dicom = pydicom.dcmread(dicoms[0])
#contours = get_contours(rt_dicom)

def normalization_hu(images):
    '''
    normalization to 0-255. and CT value from -1000 to 800 HU
    :param images:
    :return: normalized image
    '''
    MIN = -1000
    MAX = 800
    images = (images - MIN) / (MAX - MIN)
    images[images > 1] = 1.
    images[images < 0] = 0.
    return images * 255


def plot_ct_scan(scan):
    '''
    plot a few more images of the slices
    :param scan:
    :return:
    '''
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(30, 30))  # set 7 for all images
    # print('shape[0]', scan.shape[0])
    for i in range(0, scan.shape[0], 5):
        for j in range(4):
            # plots[int(i / 20), j].axis('off')
            plots[int(i / 20), j].imshow(scan[i], cmap=plt.cm.bone)
    plt.show()


if __name__ == '__main__':
    # set the path
    total_data_path = ""
    total_data_path_val = "D:\\Newdatasetandcodes\AAPM2019\\RMTCmodefieddataset\\Train"

    images_output_path = ""
    images_output_path_val = "D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\AAPM2019out"

    cases = [os.path.join(total_data_path_val, name)
             for name in sorted(os.listdir(total_data_path_val)) if
             os.path.isdir(os.path.join(total_data_path_val, name))]
    print('Patient number: ', len(cases))

    # # train sets
    for c in cases:
        # for folder_flag in range(3):
        print('c index:', cases.index(c))
        folder_flag = cases.index(c)
        print('C: ', os.path.basename(os.path.dirname(c)))
        images, masks = read_data(c, 'train_set', folder_flag, images_output_path_val)

    plot_ct_scan(images)
#%%  Convert AAPM test dataset into nifty from dicom
import os
import dicom2nifti
dicom_directory='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\testaapm2019'
output_folder='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\testaapm2019nifty'
oslistdir=os.listdir(dicom_directory)
for ii in oslistdir:
    print(os.path.join(dicom_directory,ii))
    cur_path=os.path.join(output_folder,ii)
    os.mkdir(cur_path)
    dicom2nifti.convert_directory(os.path.join(dicom_directory,ii), cur_path, compression=True, reorient=True) 
    
#dicom_directory='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\Train\\LCTSC-Train-S1-001\\11-16-2003-RTRCCTTHORAX8FLow Adult-39664\\0-CTRespCT-cran  3.0  B30s  50 Ex-63530'
#output_folder='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\Train\\LCTSC-Train-S1-001\\11-16-2003-RTRCCTTHORAX8FLow Adult-39664\\testdataset'
#dicom2nifti.convert_directory(dicom_directory, output_folder, compression=True, reorient=True)    
    
#%% Convert test images into  nfity format
import nibabel as nib
import cv2
import numpy as np
import os
#result_array = np.empty((512, 512,3))
#
#for line in data_array:
#    result = do_stuff(line)
#    result_array = np.append(result_array, [result], axis=0)
path1='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\savetestimages'
oslist=os.listdir(path1)
#train_ids = next(os.walk(path1+"Patient"))[1]
#train_ids = os.walk(path1+"train_24")
#images=next(train_ids)
#for ii in oslist:
#    print(os.path.join(data_pathnew,ii))
#    list2=os.path.join(data_pathnew,ii)
#    for jj in list2:
#        print(jj)
#        file=os.path.join(data_pathnew,ii)
#        print(os.path.join(data_pathnew,ii))
import natsort
import matplotlib.pyplot as plt
im_height=512
im_width=512
from tqdm import tqdm
for i, volume in enumerate(oslist):
    print(i)
    print(volume)
    cur_path = os.path.join(path1, volume)
    files=natsort.natsorted(os.listdir(cur_path))
#    files1=os.path.join(cur_path,files)
#    files11=natsort.natsorted(os.listdir(files1))
    #train_ids = next(os.walk(path1+"images"))
    X_train = np.zeros((len(files),im_height, im_width), dtype=np.uint8)
    for n, id_ in tqdm(enumerate(files), total=len(files)):
        print(n)
        print(id_)
        img=cv2.imread(os.path.join(cur_path,id_))
        print(img.shape)
        #img=np.swapaxes(img,0,2)
        #img1=img.transpose(1, 2, 0)
        x = np.array(img)
        X_train[n] = x
        ff=np.swapaxes(X_train,0,2)
    i=i+24
    ex="D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\testaapmnib/Patient_"+str(i)+".nii.gz"    
    img = nib.load(ex)
    im1=np.array(img.get_affine())
    new_image = nib.Nifti1Image(np.asarray(ff,dtype="uint8" ), affine=im1)
    nib.save(new_image,'D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\testsegthornii1mask/Patient_'+str(i)+'_GT.nii.gz')

#fff=img.transpose(1, 2, 0)
#ff1=fff[:,:,1]
#%%
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
saveslices="D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\Trainingmaskcolr"
dir_data = "D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset"
dir_seg = dir_data + "/Trainingmask/"
#dir_img = dir_data + "/testimages/"
#def imgcolor(img,color,shape):
#    img=img.reshape((-1,3))
#    img=np.multiply(img, color)
#    img=img.reshape((shape[0],shape[1],3))
#    return img
#check labels of all classes
import natsort
directorylist=os.listdir(dir_seg)
directorylist.sort()
d=1
for segm in natsort.natsorted(directorylist,reverse=False):
    seg = cv2.imread(dir_seg + segm ) # (360, 480, 3)
    #seg=cv2.resize(seg,(256,256))
    Totimagemask= np.zeros([512,512],dtype=np.uint8)
    Totimagemask=  np.expand_dims(Totimagemask, 2) 
    Totimagemask = np.repeat(Totimagemask, 3, axis=2) # give the mask the same shape as your image
    seg1=seg[:,:,2]
    idx=(seg1==1)
    if seg1[idx].any():
        image= np.zeros([512,512],dtype=np.uint8)
        image[idx]=seg1[idx]
        mask =  np.expand_dims(image, 2) 
        mask = np.repeat(mask, 3, axis=2) # give the mask the same shape as your image
        colors = {"red": [255, 0, 0], "blue": [255, 0, 0]} # a dictionary for your colors, experiment with the values
        colored_mask = np.multiply(mask, colors["red"])  # broadcast multiplication (thanks to the multiplication by 0, you'll end up with values different from 0 only on the relevant channels and the right regions)
        imager = colored_mask # element-wise sum (sinc img and mask have the same shape)
        imager=np.array(imager,dtype=np.uint8 )
        Totimagemask =Totimagemask +imager
#        plt.imshow(np.float64(Totimagemask))
#        plt.show()
    idx=(seg1==2)
    if seg1[idx].any():
        image1= np.zeros([512,512],dtype=np.uint8)
        image1[idx]=seg1[idx]
        mask1 =  np.expand_dims(image1, 2) 
        mask1 = np.repeat(mask1, 3, axis=2) # give the mask the same shape as your image
        colors = {"red": [255, 0, 0], "blue": [0, 255, 0]} # a dictionary for your colors, experiment with the values
        colored_mask1 = np.multiply(mask1, colors["blue"])  # broadcast multiplication (thanks to the multiplication by 0, you'll end up with values different from 0 only on the relevant channels and the right regions)
        image1 = colored_mask1 # element-wise sum (sinc img and mask have the same shape)
        Totimagemask =Totimagemask +image1
        image1=np.array(image1,dtype=np.uint8 )
#        plt.imshow(np.float64(image1))
#        plt.show()
    
    idx=(seg1==3)
    if seg1[idx].any():
        image2= np.zeros([512,512],dtype=np.uint8)
        image2[idx]=seg1[idx]
        image2[idx]=seg1[idx]
        mask2 =  np.expand_dims(image2, 2) 
        mask2 = np.repeat(mask2, 3, axis=2) # give the mask the same shape as your image
        colors = {"red": [0, 0, 255], "blue": [255, 0, 0],"yellow":[0,0,255]} # a dictionary for your colors, experiment with the values
        colored_mask2 = np.multiply(mask2, colors["yellow"])  # broadcast multiplication (thanks to the multiplication by 0, you'll end up with values different from 0 only on the relevant channels and the right regions)
        image2 = colored_mask2 # element-wise sum (sinc img and mask have the same shape)
        image2=np.array(image2,dtype=np.uint8 )
        Totimagemask =Totimagemask +image2
    idx=(seg1==4)
    if seg1[idx].any():
        image2= np.zeros([512,512],dtype=np.uint8)
        image2[idx]=seg1[idx]
        image2[idx]=seg1[idx]
        mask2 =  np.expand_dims(image2, 2) 
        mask2 = np.repeat(mask2, 3, axis=2) # give the mask the same shape as your image
        colors = {"red": [0, 0, 255], "blue": [255, 0, 0],"yellow1":[0,255,255]} # a dictionary for your colors, experiment with the values
        colored_mask2 = np.multiply(mask2, colors["yellow1"])  # broadcast multiplication (thanks to the multiplication by 0, you'll end up with values different from 0 only on the relevant channels and the right regions)
        image2 = colored_mask2 # element-wise sum (sinc img and mask have the same shape)
        image2=np.array(image2,dtype=np.uint8 )
        Totimagemask =Totimagemask +image2
    idx=(seg1==5)
    if seg1[idx].any():
        image2= np.zeros([512,512],dtype=np.uint8)
        image2[idx]=seg1[idx]
        image2[idx]=seg1[idx]
        mask2 =  np.expand_dims(image2, 2) 
        mask2 = np.repeat(mask2, 3, axis=2) # give the mask the same shape as your image
        colors = {"red": [0, 0, 255], "blue": [255, 0, 0],"yellow11":[255,165,0]} # a dictionary for your colors, experiment with the values
        colored_mask2 = np.multiply(mask2, colors["yellow11"])  # broadcast multiplication (thanks to the multiplication by 0, you'll end up with values different from 0 only on the relevant channels and the right regions)
        image2 = colored_mask2 # element-wise sum (sinc img and mask have the same shape)
        image2=np.array(image2,dtype=np.uint8 )
        Totimagemask =Totimagemask +image2
    idx=(seg1==6)
    if seg1[idx].any():
        image2= np.zeros([512,512],dtype=np.uint8)
        image2[idx]=seg1[idx]
        image2[idx]=seg1[idx]
        mask2 =  np.expand_dims(image2, 2) 
        mask2 = np.repeat(mask2, 3, axis=2) # give the mask the same shape as your image
        colors = {"red": [0, 0, 255], "blue": [255, 0, 0],"yellow1r":[128,128,128]} # a dictionary for your colors, experiment with the values
        colored_mask2 = np.multiply(mask2, colors["yellow1r"])  # broadcast multiplication (thanks to the multiplication by 0, you'll end up with values different from 0 only on the relevant channels and the right regions)
        image2 = colored_mask2 # element-wise sum (sinc img and mask have the same shape)
        image2=np.array(image2,dtype=np.uint8 )
        Totimagemask =Totimagemask +image2
    idx=(seg1==7)
    if seg1[idx].any():
        image2= np.zeros([512,512],dtype=np.uint8)
        image2[idx]=seg1[idx]
        image2[idx]=seg1[idx]
        mask2 =  np.expand_dims(image2, 2) 
        mask2 = np.repeat(mask2, 3, axis=2) # give the mask the same shape as your image
        colors = {"red": [0, 0, 255], "blue": [255, 0, 0],"yellow1b":[128,0,0]} # a dictionary for your colors, experiment with the values
        colored_mask2 = np.multiply(mask2, colors["yellow1b"])  # broadcast multiplication (thanks to the multiplication by 0, you'll end up with values different from 0 only on the relevant channels and the right regions)
        image2 = colored_mask2 # element-wise sum (sinc img and mask have the same shape)
        image2=np.array(image2,dtype=np.uint8 )
        Totimagemask =Totimagemask +image2
    idx=(seg1==8)
    if seg1[idx].any():
        image2= np.zeros([512,512],dtype=np.uint8)
        image2[idx]=seg1[idx]
        image2[idx]=seg1[idx]
        mask2 =  np.expand_dims(image2, 2) 
        mask2 = np.repeat(mask2, 3, axis=2) # give the mask the same shape as your image
        colors = {"red": [0, 0, 255], "blue": [255, 0, 0],"yellow1g":[128,128,0]} # a dictionary for your colors, experiment with the values
        colored_mask2 = np.multiply(mask2, colors["yellow1g"])  # broadcast multiplication (thanks to the multiplication by 0, you'll end up with values different from 0 only on the relevant channels and the right regions)
        image2 = colored_mask2 # element-wise sum (sinc img and mask have the same shape)
        image2=np.array(image2,dtype=np.uint8 )
        Totimagemask =Totimagemask +image2
#        plt.imshow(np.float64(image2))
#        plt.show()
    mycardiumt=Totimagemask
#        contents[:,:,itmp] = mycardiumt
#        itmp = itmp +1
#        res = np.sum(contents,axis=2)
##        plt.show()
##        plt.imshow(res,cmap=plt.cm.gray)
#        print(itmp)
        #filename = "slices_mask_%d.png"%d
    filename = "%d.png"%d
        #cv2.imwrite(saveslices+'\\'+filename, total * (255/3)) # save images
    cv2.imwrite(os.path.join(saveslices,segm.split('.')[0]+'.png'),mycardiumt) # save images
        #$cv2.imwrite(saveslices+'\\'+filename, res) # save images
    d+=1
#%%
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import natsort
saveslices="D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\overlaping"
dir_data = "D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset"
dir_seg = dir_data + "/Trainingmaskcolr/"
dir_img = dir_data + "/Trainingimages/"
#def imgcolor(img,color,shape):
#    img=img.reshape((-1,3))
#    img=np.multiply(img, color)
#    img=img.reshape((shape[0],shape[1],3))
#    return img
#check labels of all classes
directorylist=os.listdir(dir_img)
directorylist.sort()
d=1
for segm in natsort.natsorted(directorylist, reverse=False):
    seg = cv2.imread(dir_seg+segm.split('.')[0]+'_label'+'.png') # (360, 480, 3)
    imagedata=cv2.imread(dir_img+segm)
    #imagedata1=cv2.resize(imagedata,(256,256))
    #img = resizeimage.resize_cover(imagedata, [224, 224])
    fullimage= cv2.add(imagedata, seg )
    filename = "%d.png"%d
        #cv2.imwrite(saveslices+'\\'+filename, total * (255/3)) # save images
    cv2.imwrite(saveslices+'\\'+filename,fullimage) # save images
        #$cv2.imwrite(saveslices+'\\'+filename, res) # save images
    d+=1     
    
    
    
#%%
import nibabel as nib
import cv2
import numpy as np
import os
#result_array = np.empty((512, 512,3))
#
#for line in data_array:
#    result = do_stuff(line)
#    result_array = np.append(result_array, [result], axis=0)
savefile='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\savetestimages'
path1='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\testdata'
oslist=os.listdir(path1)
#train_ids = next(os.walk(path1+"Patient"))[1]
#train_ids = os.walk(path1+"train_24")
#images=next(train_ids)
#for ii in oslist:
#    print(os.path.join(data_pathnew,ii))
#    list2=os.path.join(data_pathnew,ii)
#    for jj in list2:
#        print(jj)
#        file=os.path.join(data_pathnew,ii)
#        print(os.path.join(data_pathnew,ii))
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

import natsort
import matplotlib.pyplot as plt
im_height=512
im_width=512
from tqdm import tqdm
for i, volume in enumerate(oslist):
    print(i)
    print(volume)
    cur_path = os.path.join(path1, volume)
    files=natsort.natsorted(os.listdir(cur_path))[0]
    files1=os.path.join(cur_path,files)
    files11=natsort.natsorted(os.listdir(files1))
    #train_ids = next(os.walk(path1+"images"))
    ff=files1.split('\\')[0:6]
    ff[5]
    for n, id_ in tqdm(enumerate(files11), total=len(files11)):
        print(n)
        print(id_)
        img=np.load(os.path.join(files1,id_))
        print(img.shape)
        #img=np.swapaxes(img,0,2)
        x = np.array(img)[0,:,:]
        #ggg=savefile+ff[5]
        #fff=os.mkdir(os.path.join(savefile,ff[5])) 
        save_path=os.path.join(savefile,ff[5])
        createFolder(save_path)
        cv2.imwrite(os.path.join(save_path,ff[5]+'_'+id_.split('.')[0]+'.png'),x)
#os.mkdir(os.path.join(savefile,ff[5]))      
#%%        
        
#%%
data_path='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\testsegthornii1'
directeoru=os.listdir(data_path)
for ii in directeoru:
    print(os.path.join(data_path,ii))
    img = nib.load(os.path.join(data_path,ii))
    #img = nib.load(data_path)
    data = img.get_fdata()
    print(data.shape)
#%% 
data_path='D:\\Newdatasetandcodes\\AAPM2019\\RMTCmodefieddataset\\testaapmnib'
directeoru=os.listdir(data_path)
for ii in directeoru:
    print(os.path.join(data_path,ii))
    img = nib.load(os.path.join(data_path,ii))
    #img = nib.load(data_path)
    data = img.get_fdata()
    print(data.shape)    
    
    
    
#%%    
import os, sys, glob

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
def get_masks(contours, shape, scans):
    '''
    convert RT Structure to label map mask
    :param contours
    :param shape:
    :param scans:
    :return:
    '''
    z = [np.around(s.ImagePositionPatient[2], 1) for s in scans]
    pos_row = scans[0].ImagePositionPatient[1]
    # print(scans[0].ImagePositionPatient)
    spacing_row = scans[0].PixelSpacing[1]
    pos_column = scans[0].ImagePositionPatient[0]
    spacing_column = scans[0].PixelSpacing[0]

    mask = np.zeros(shape, dtype=np.float32)
    for con in contours:
        num = ROI_ORDER.index(con['name']) + 1
        for c in con['contours']:
            nodes = np.array(c).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = z.index(np.around(nodes[0, 2], 1))
            # print('z_index: ', z_index)
            r = (nodes[:, 1] - pos_row) / spacing_row
            c = (nodes[:, 0] - pos_column) / spacing_column
            rr, cc = polygon(r, c)
            mask[z_index, rr, cc] = int(num)
    # print('mask shape: ', mask[0], 'num shape', num)
    return mask
# import code end


def normalization_hu(images):
    '''
    normalization to 0-255. and CT value from -1000 to 800 HU
    :param images:
    :return: normalized image
    '''
    MIN = -1000
    MAX = 800
    images = (images - MIN) / (MAX - MIN)
    images[images > 1] = 1.
    images[images < 0] = 0.
    return images * 255


def plot_ct_scan(scan):
    '''
    plot a few more images of the slices
    :param scan:
    :return:
    '''
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(30, 30))  # set 7 for all images
    # print('shape[0]', scan.shape[0])
    for i in range(0, scan.shape[0], 5):
        for j in range(4):
            # plots[int(i / 20), j].axis('off')
            plots[int(i / 20), j].imshow(scan[i], cmap=plt.cm.bone)
    plt.show()

# some code imported from http://aapmchallenges.cloudapp.net/forums/3/2/
def get_contours(structure):
    '''
    get structure and reference information from RT Structure DICOM file
    :param structure: RT Structure file
    :return: contour
    '''
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
        contour['name'] = structure.StructureSetROISequence[i].ROIName
        assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
    return contours


ROI_ORDER = ['SpinalCord', 'Lung_R', 'Lung_L', 'Heart', 'Esophagus']

mask_file="D:\\Newdatasetandcodes\\OARnewdataset\\Mask\\1-.simplified-62948\\000000.dcm"

for (dirpath, dirnames, filenames) in os.walk(mask_file):
    listOfFiles1 += [os.path.join(dirpath, file) for file in filenames]
for i in dirs:
        # get the number of current path
        dicoms = glob.glob(os.path.join(root + '\\' + i, '*.dcm'))
        print(dirs)
rt_dicom = pydicom.dcmread(mask_file)
contours = get_contours(rt_dicom)
def get_masks(contours, shape, scans):
    '''
    convert RT Structure to label map mask
    :param contours
    :param shape:
    :param scans:
    :return:
    '''
    z = [np.around(s.ImagePositionPatient[2], 1) for s in scans]
    pos_row = scans[0].ImagePositionPatient[1]
    # print(scans[0].ImagePositionPatient)
    spacing_row = scans[0].PixelSpacing[1]
    pos_column = scans[0].ImagePositionPatient[0]
    spacing_column = scans[0].PixelSpacing[0]

    mask = np.zeros(shape, dtype=np.float32)
    for con in contours:
        num = ROI_ORDER.index(con['name']) + 1
        for c in con['contours']:
            nodes = np.array(c).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = z.index(np.around(nodes[0, 2], 1))
            # print('z_index: ', z_index)
            r = (nodes[:, 1] - pos_row) / spacing_row
            c = (nodes[:, 0] - pos_column) / spacing_column
            rr, cc = polygon(r, c)
            mask[z_index, rr, cc] = int(num)
    # print('mask shape: ', mask[0], 'num shape', num)
    return mask

masks = get_masks(contours, images.shape, scans)
mask_outp="D:\\Newdatasetandcodes\\OARnewdataset\\Mask\\maskfile"
for msk in range(0, len(masks)):
    # save as png
    # labels_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
    # print('base name', os.path.basename(root))
    folder = os.path.exists(mask_outp + '\\train1_' + "/masks")
    if not folder:
        os.mkdir(mask_outp + '\\train1_')
        os.mkdir(mask_outp + '\\train1_' + '/masks')
    if msk == 0 or msk == len(images) - 1:
        continue
    # cur_mask = masks[msk - 1: msk + 2, :, :].astype('uint8')
    # cur_mask = masks[msk, :, :].astype('uint8')
    np.save(mask_outp + '\\train1_' +'\\masks\\' + str(msk) + '_mask.npy', masks[msk])


for img in range(0, len(images)):
    total_imgs = []
    folder = os.path.exists(images_output_path + '\\train_' + '/images')
    if not folder:
        os.mkdir(images_output_path + '\\train_' )
        os.mkdir(images_output_path + '\\train_' + '/images')
    if img == 0 or img == len(images) - 1:
        continue
    # print(img)
    # 2.5D data, using adjacent 3 images
    cur_img = images[img - 1:img + 2, :, :].astype('uint8')
    np.save(images_output_path + '\\train_' + '\\images\\' + str(img) + '_image.npy', cur_img)


path="D:\\Newdatasetandcodes\\OARnewdataset\\imageoutpath\train_\\images\\45_image.npy"
img = np.load('D:\\Newdatasetandcodes\\OARnewdataset\\imageoutpath\train_\\images\\45_image.npy')
plt.figure(figsize=(20,10))
for j in range(2):
    q = j+1
    j = j+1
    img = np.load(path+'images/' + str(45) + '_image.npy').transpose(1, 2, 0)
    img_mask = np.load(path+'masks/' + str(45) +'_mask.npy')
    
    plt.subplot(1,2*3,q*2-1)
    plt.imshow(img)
    plt.subplot(1,2*3,q*2)
    plt.imshow(img_mask)
plt.show()


total_data_path = "D:\\Newdatasetandcodes\\OARnewdataset\\Train"

dirtectorylist=os.listdir(total_data_path)
for i in dirtectorylist[0]:
    dicoms = glob.glob(os.path.join(total_data_path + '\\' + i, '*.dcm')) 

for i in os.scandir(total_data_path):
    if i.is_file():
        print('File: ' + i.path)
    elif i.is_dir():
        print('Folder: ' + i.path)

import os

dirname = total_data_path.split(os.path.sep)[-1]  

import os

# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(total_data_path):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        print(len(path) * '---', file)

for i in path[5]:
 
for i in dirtectorylist:
    print(i)
for (dirpath, dirnames, filenames) in os.walk(total_data_path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]
for i in dirs:
        # get the number of current path
        dicoms = glob.glob(os.path.join(root + '\\' + i, '*.dcm'))
        print(dirs)
    
    
    
# Get the list of all files in directory tree at given path
#listOfFiles = list()
#for (dirpath, dirnames, filenames) in os.walk(total_data_path):
#    listOfFiles += [os.path.join(dirpath, file) for file in filenames]
#
#
#'''
#    For the given path, get the List of all files in the directory tree 
#'''
#def getListOfFiles(dirName):
#    # create a list of file and sub directories 
#    # names in the given directory 
#    listOfFile = os.listdir(dirName)
#    allFiles = list()
#    # Iterate over all the entries
#    for entry in listOfFile:
#        # Create full path
#        fullPath = os.path.join(dirName, entry)
#        # If entry is a directory then get the list of files in this directory 
#        if os.path.isdir(fullPath):
#            allFiles = allFiles + getListOfFiles(fullPath)
#        else:
#            allFiles.append(fullPath)
#                
#    return allFiles        
# 
# 
#def main():
#    
#    dirName = "D:\\Newdatasetandcodes\\OARnewdataset\\Train";
#    
#    # Get the list of all files in directory tree at given path
#    listOfFiles = getListOfFiles(dirName)
#    
#    # Print the files
#    for elem in listOfFiles:
#        print(elem)
# 
#    print ("****************")
#    
#    # Get the list of all files in directory tree at given path
#    listOfFiles = list()
#    for (dirpath, dirnames, filenames) in os.walk(dirName):
#        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
#        
#        
#    # Print the files    
#    for elem in listOfFiles:
#        print(elem)    
#        
#        
#        
#        
#if __name__ == '__main__':
#    main()


total_data_path_val = "D:\\Newdatasetandcodes\\OARnewdataset\\testdata"
images_output_path = "D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\outputtrainimages\\"
for root, dirs, file in os.walk(total_data_path_val):
    for i in dirs:
        # get the number of current path
        dicoms = glob.glob(os.path.join(root + '\\' + i, '*.dcm'))
        # read RT structure files
#        if train_flag == 'train_set' and 'simplified' in os.path.split(i)[1]:
#            rt_dicom = pydicom.dcmread(dicoms[0])
#            contours = get_contours(rt_dicom)
            # read dicoms
        if len(dicoms) > 0:
            scans = [None] * len(dicoms)
            for i in range(len(dicoms)):
                scans[i] = pydicom.dcmread(dicoms[i])
                print(scans[i].pixel_array.shape)
                plt.imshow(scans[i].pixel_array, cmap=plt.cm.bone)
                plt.show()

                # ['ImagePositionPatient', 'PatientPosition', 'PositionReferenceIndicator']
                print(scans[i].dir('position'))
                # print(scans[i].ImagePositionPatient)
                # sort by z
                scans.sort(key=lambda z: float(z.ImagePositionPatient[2]))
                #
                images = np.stack([s.pixel_array for s in scans])
                images = images.astype(np.int16)
                print(images.shape)

            # HU
            for s_num in range(len(scans)):
                intercept = scans[s_num].RescaleIntercept
                slope = scans[s_num].RescaleSlope
                # print('inter:', intercept, 'slope', slope)

                if slope != 1:
                    images[s_num] = slope * images[s_num].astype(np.float64)
                    images[s_num] = images[s_num].astype(np.int16)
                    images[s_num] += np.int16(intercept)
                    # plot HU for every image
                    # plt.hist(images[s_num].flatten(), bins=80, color='c')
                    # plt.show()

    # normalization HU
    images = normalization_hu(images)
    # plt.hist(images[s_num].flatten(), bins=80, color='c')
    # plt.show()

images_output_path="D:\\Newdatasetandcodes\\OARnewdataset\\imageoutpath"
# save images as npy
for img in range(0, len(images)):
    total_imgs = []
    folder = os.path.exists(images_output_path + '\\train_' + '/images')
    if not folder:
        os.mkdir(images_output_path + '\\train_' )
        os.mkdir(images_output_path + '\\train_' + '/images')
    if img == 0 or img == len(images) - 1:
        continue
    # print(img)
    # 2.5D data, using adjacent 3 images
    cur_img = images[img - 1:img + 2, :, :].astype('uint8')
    np.save(images_output_path + '\\train_' + '\\images\\' + str(img) + '_image.npy', cur_img)
    


#masks = get_masks(contours, images.shape, scans)
#    # save masks as npy
#for msk in range(0, len(masks)):
#    # save as png
#    # labels_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
#    # print('base name', os.path.basename(root))
#    folder = os.path.exists(out_path + '\\train_' + "/masks")
#    if not folder:
#        os.mkdir(images_output_path + '\\train_' + "/masks")
#    if msk == 0 or msk == len(images) - 1:
#        continue
#    # cur_mask = masks[msk - 1: msk + 2, :, :].astype('uint8')
#    # cur_mask = masks[msk, :, :].astype('uint8')
#    np.save(images_output_path + '\\train_' + '/masks/' + str(msk) + '_mask.npy', masks[msk])


    # future work for semi-supervised learning
    # elif train_flag == 'semi_supervised_set':
    #     # save images as png
    #     # need to be un comment
    #     for img in range(0, len(images)):
    #         # images_output_path = os.path.join(os.path.join(root, os.path.pardir), os.path.pardir)
    #         folder = os.path.exists(out_path + '/test_' + str(floder_flag) + "/images")
    #         if not folder:
    #             os.mkdir(out_path + "/images")
    #         else:
    #             np.save(out_path + '/train_' + str(floder_flag) + '/images/' + str(img) + '_image.npy', cur_img)
    #     return images

for root, dirs, file in os.walk(path):
    for i in dirs:
#     # get the number of current path
#            dicoms = glob.glob(os.path.join(root + '/' + i, '*.dcm'))
#
#            # read RT structure files
#            if train_flag == 'train_set' and 'simplified' in os.path.split(i)[1]:
#                rt_dicom = pydicom.dcmread(dicoms[0])
#                contours = get_contours(rt_dicom)




total_data_path = "D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\train"
#total_data_path_val = "D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\valid"
images_output_path = "D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\outputtrainimages"
images_output_path_val = "D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\outputvalidimages"
cases = [os.path.join(total_data_path, name)
        for name in sorted(os.listdir(total_data_path)) if
        os.path.isdir(os.path.join(total_data_path, name))]
print('Patient number: ', len(cases))
for c in cases:
    print('c index:', cases.index(c))
    folder_flag = cases.index(c)
    print('C: ', os.path.basename(os.path.dirname(c)))
    

    # # train sets
    for c in cases:
        # for folder_flag in range(3):
        print('c index:', cases.index(c))
        folder_flag = cases.index(c)
        print('C: ', os.path.basename(os.path.dirname(c)))
        images, masks = read_data(c, 'train_set', folder_flag, images_output_path_val)

    plot_ct_scan(images)





if __name__ == '__main__':
    # set the path
    total_data_path = "D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\train"
    total_data_path_val = "D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\valid"

    images_output_path = "D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\outputtrainimages"
    images_output_path_val = "D:\\Newdatasetandcodes\\Maincodes2.5d\\SeGThtraindata\\outputvalidimages"

    cases = [os.path.join(total_data_path_val, name)
             for name in sorted(os.listdir(total_data_path_val)) if
             os.path.isdir(os.path.join(total_data_path_val, name))]
    print('Patient number: ', len(cases))

    # # train sets
    for c in cases:
        # for folder_flag in range(3):
        print('c index:', cases.index(c))
        folder_flag = cases.index(c)
        print('C: ', os.path.basename(os.path.dirname(c)))
        images, masks = read_data(c, 'train_set', folder_flag, images_output_path_val)

    plot_ct_scan(images)