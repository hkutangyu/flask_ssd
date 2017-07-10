from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
from PIL import Image
import math

# global variables
target_img_num = 250
width_threshold = 900
aug_db_path = '/var/www/flask_ssd/flask_ssd/tmp'

# define image generator
datagen = ImageDataGenerator(
    rotation_range=270,
    width_shift_range=0.01,
    height_shift_range=0.01,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)


def aug_folder(org_folder_path):

    # get folder name
    folder_name = org_folder_path.split('/')[-1]

    # calculate how many images in 'org_folder_path'
    num_images = len(sum([i[2] for i in os.walk(org_folder_path)], []))

    # beside num_images, how many images we need
    num_images_left = target_img_num - num_images

    images_per_aug = math.ceil(num_images_left / num_images)

    if images_per_aug <= 0:
        images_per_aug = 1

    total_num_images = 0

    for file_path in os.listdir(org_folder_path):
        img = load_img(os.path.join(org_folder_path, file_path))

        # if img's width greater than 900, we resize the image
        if img.size[0] > width_threshold:
            ratio = img.size[0] / img.size[1]
            height_threshold = int(width_threshold / ratio)
            img = img.resize((width_threshold, height_threshold), Image.ANTIALIAS)
        x = img_to_array(img)  # this is a Numpy array with shape (3, width_threshold, height_threshold)
        x = x.reshape((1, ) + x.shape)  # this is a Numpy array with shape (1, 3, width_threshold, height_threshold)

        i = 0

        if total_num_images >= num_images_left:
            break

        for batch in datagen.flow(x,
                                  batch_size=1,
                                  save_to_dir=org_folder_path,
                                  save_prefix='temp',
                                  save_format='jpg'):
            i += 1
            total_num_images += 1
            if i >= images_per_aug or total_num_images >= num_images_left:
                break

    dst_i = 1
    for file_path in os.listdir(org_folder_path):
        if os.path.isfile(os.path.join(org_folder_path, file_path)):
            #print("process {0}".format(file_path))
            os.rename("{0}/{1}".format(org_folder_path, file_path),
                      "{0}/{1}_{2}.jpg".format(org_folder_path, folder_name, dst_i))
            dst_i += 1

#aug_folder('/var/www/flask_ssd/flask_ssd/tmp/ERKE')

for f_name in os.listdir(aug_db_path):
    print(f_name)
    aug_folder(os.path.join(aug_db_path, f_name))
