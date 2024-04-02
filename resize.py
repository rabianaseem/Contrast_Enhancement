import tensorflow as tf
import numpy as np
import random, cv2, operator, os
from PIL import Image

dir="/home/user1/Downloads/pytorch-deep-photo-enhancer-master/images_LR/input/"

def resize_image(dir, max_length=512):
        subdirs = [os.path.join(dir, dI) for dI in os.listdir(dir) if os.path.isdir(os.path.join(dir, dI))]
        for subdir in subdirs:
                for file in os.listdir(subdir):
                        image = Image.open(subdir + '/' + file)
                        print(image.size)
                        max_size = np.argmax(image.size)
                        width, height = image.size
                        factor = max_length / image.size[max_size]
                        new_image = image.resize((int(factor * width), int(factor * height)), Image.ANTIALIAS)
                        image.close()
                        new_image.save(subdir + '/' + file)


if __name__ =="__main__":

        resize_image("/home/user1/Downloads/pytorch-deep-photo-enhancer-master/images_LR/input/Training1")
        resize_image("/home/user1/Downloads/pytorch-deep-photo-enhancer-master/images_LR/input/Training2")
        resize_image("/home/user1/Downloads/pytorch-deep-photo-enhancer-master/images_LR/input/Testing")

        resize_image("/home/user1/Downloads/pytorch-deep-photo-enhancer-master/images_LR/Reference/Training1")
        resize_image("/home/user1/Downloads/pytorch-deep-photo-enhancer-master/images_LR/Reference/Training2")
        resize_image("/home/user1/Downloads/pytorch-deep-photo-enhancer-master/images_LR/Reference/Testing")


