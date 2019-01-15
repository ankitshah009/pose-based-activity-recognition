import cv2
import os
import numpy as np

def potion_transform(path):
    images = []

    for filename in sorted(os.listdir(path)):
        img = cv2.imread(os.path.join(path,filename), cv2.IMREAD_GRAYSCALE)

        # Transform image to monochrome
        if img is not None:
            potion_t = img.copy()
            potion_t[np.where(potion_t > [0])] = [255]
            potion_t = cv2.cvtColor(potion_t, cv2.COLOR_GRAY2BGR)

            images.append(potion_t)

    pose_list = []
    agg_image = images[0]
    agg_image[np.where(True)] = [0, 0, 0]
    canvas_split = len(images)//2 + 1
    r = g = True
    b = False

    for idx, img in enumerate(images):

        # Modulate ratio to get blend of red-green and green-blue
        ratio = (idx % canvas_split)/canvas_split
        if idx >= canvas_split:
            r = False
            b = True
            ratio = 1 - ratio

        # Normalize the pixel intensities - no. of frames
        # img[np.where((img == [255, 255, 255]).all(axis=2))] = [b*(1 - ratio)*255/canvas_split, g*ratio*255/canvas_split, r*(1-ratio)*255/canvas_split]
        img[np.where((img == [255, 255, 255]).all(axis=2))] = [b*(1 - ratio)*255, g*ratio*255, r*(1-ratio)*255]
        pose_list.append(img)

        agg_image = cv2.add(agg_image, img)
        name = "/data/MM1/aps1/aniru/aniru/actionRecognition/output/agg_image_" + str(idx) + ".png" 
        cv2.imwrite(name, img)

    return images, pose_list, agg_image

im_list, pose_list, agg_image = potion_transform("/data/MM1/aps1/aniru/aniru/dataset/output_heatmaps_folder/v_ApplyEyeMakeup_g08_c02/")
cv2.imwrite("/data/MM1/aps1/aniru/aniru/actionRecognition/output/agg_image.png", agg_image)
