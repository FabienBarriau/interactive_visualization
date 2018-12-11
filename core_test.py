from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

import GUI_utils as gui

MINI_IMG_SHAPE = [48, 48]

file = "image/"

names_list = list()
mini_img = list()

# Read image folder
for name in os.listdir(file):
    extension = name.split(".")[1]
    if extension.lower() in ["jpg", "png"]:
        names_list += [name]
        im = Image.open(file + name)
        im = im.resize(MINI_IMG_SHAPE, Image.NEAREST)
        mini_img += [np.asarray(im)]

result = np.asarray([[1, 2], [1, 4], [3, 4], [4, 5], [3, 1], [5, 3]])

fig, ax = gui.image_plot(X_2D=result, images=mini_img, figsize=MINI_IMG_SHAPE)
print(ax.axis)
# click = gui.CustomPick(ax, func=gui.display_original_img, file=file, names_list=names_list)
click = gui.CustomPick(ax, func=gui.display_RenameApp, file=file, names_list=names_list)
plt.show(fig)
