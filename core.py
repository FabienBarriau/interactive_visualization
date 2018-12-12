from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
from argparse import ArgumentParser
import numpy as np
import os
import matplotlib.pyplot as plt

import GUI_utils as gui

MINI_IMG_SHAPE = [48, 48]

#Choose File
#Choose visualisation method PCA or TSNE, by default it will be PCA if you put anything different of TSNE
#Choose model VGG16 or VGG19, by default it will be VGG16 if you put anything different of VGG19

parser = ArgumentParser()
parser.add_argument('--file',
                    dest='file', help='file where there are images',
                    metavar='FILE', required=True)
parser.add_argument('--visu',
                    dest='visu_method', help='PCA or TSNE',
                    metavar='VISU_METHOD', default="PCA")
parser.add_argument('--model',
                    dest='model', help='VGG16 or VGG19',
                    metavar='MODEL', default="VGG16")

options = parser.parse_args()

#Load img [224,224] to calculate their VGG16 last convolution layers's encoding
#Load mini_img MINI_IMG_SHAPE to plot them in the final plot.
#Load img names

names_list = list()
img_list = list()
mini_img = list()

for name in os.listdir(options.file):
    extension = name.split(".")[1]
    if extension.lower() in ["jpg", "png"]:
        names_list.append(name)
        im = Image.open(options.file + name)
        img_list.append(np.asarray(im.resize([224, 224], Image.NEAREST)))
        mini_img.append(np.asarray(im.resize(MINI_IMG_SHAPE, Image.NEAREST)))

tensor = np.stack(img_list, axis=0)

#Deploy VGG16 model to extract encoding at the last convolutional layer
if options.model == "VGG19":
    model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None,
                                           input_shape=(224, 224, 3), pooling="max", classes=1000)
else:
    model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None,
                                       input_shape=(224, 224, 3), pooling="max", classes=1000)

encoding = model.predict(tensor, batch_size=5, verbose=1)

#Reduce dimension from 512D to 2D for visualization with PCA or TSNE
if options.visu_method == "TSNE":
    result = TSNE(n_components=2).fit_transform(encoding)
else:
    result = PCA(n_components=2).fit_transform(encoding)

fig, ax = gui.image_plot(X_2D=result, images=mini_img, figsize=MINI_IMG_SHAPE)
print(ax.axis)
click = gui.CustomPick(ax, func=gui.display_RenameApp, file=options.file, names_list=names_list)
plt.show(fig)


