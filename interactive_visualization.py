from tensorflow import keras
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from argparse import ArgumentParser
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg


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
    if (extension == "jpg") | (extension == "JPG") | (extension == "png") | (extension == "PNG"):
        names_list += [name]
        im = Image.open(options.file + name)
        im = im.resize([224, 224], Image.NEAREST)
        img_list += [np.asarray(im)]
        im = im.resize(MINI_IMG_SHAPE, Image.NEAREST)
        mini_img += [np.asarray(im)]

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

#Create scatter image plot
def scatter_image_plot(X_2D, images, figsize=MINI_IMG_SHAPE, image_zoom=1):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Interactive visualization')
    plt.axis('off')
    artists = []
    for xy, i in zip(X_2D, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2D)
    ax.autoscale()
    return fig, ax

#We want to include display information about the original image when we click on the mini image.
#There are conflict between clicking and dragging to zoom so we must create a special class
class Click():
    def __init__(self, ax, func, button=1):
        self.ax=ax
        self.func=func
        self.button=button
        self.press=False
        self.move = False
        self.c1=self.ax.figure.canvas.mpl_connect('button_press_event', self.onpress)
        self.c2=self.ax.figure.canvas.mpl_connect('button_release_event', self.onrelease)
        self.c3=self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onmove)

    def onclick(self,event):
        if event.inaxes == self.ax:
            if event.button == self.button:
                self.func(event, self.ax)

    def onpress(self,event):
        self.press=True
    def onmove(self,event):
        if self.press:
            self.move=True
    def onrelease(self,event):
        if self.press and not self.move:
            self.onclick(event)
        self.press=False; self.move=False

#There is a little app that provide tools to change the name  and show the original image.
class RenameApp():
    def __init__(self, init_name, index):
        self.init_name = init_name
        self.extension = init_name.split(".")[1]
        self.index = index

        self.window = tk.Tk()

        self.entry = tk.Entry(self.window)
        self.entry.insert(10, init_name.split(".")[0])
        self.save_name_button = tk.Button(self.window, text="Change name", command=self.change_name)
        self.restore_init_name_button = tk.Button(self.window, text="Restore init name", command=self.restore_init_name)
        self.quit_button = tk.Button(self.window, text="Quit", command=self.quit)

        self.save_name_button.pack()
        self.restore_init_name_button.pack()
        self.quit_button.pack()
        self.entry.pack()

        self.fig = plt.figure(init_name.split(".")[0])
        plt.imshow(mpimg.imread(options.file + names_list[index]))
        plt.axis('off')

    def launch(self):
        self.fig.show()
        self.window.mainloop()

    def change_name(self):
        if len(self.entry.get().split(".")) < 2:
            new_name = self.entry.get() + "." + self.extension
            os.rename(options.file + names_list[self.index], options.file + new_name)
            names_list[self.index] = new_name
            self.fig.canvas.set_window_title(self.entry.get())
        else:
            print("Error no point allowed in the entry")

    def restore_init_name(self):
        os.rename(options.file + names_list[self.index], options.file + self.init_name)
        names_list[self.index] = self.init_name
        self.fig.canvas.set_window_title(self.init_name.split(".")[0])
        self.entry.delete(first=0, last=22)
        self.entry.insert(10, self.init_name.split(".")[0])

    def quit(self):
        plt.close(self.fig)
        self.window.quit()
        self.window.destroy()

#Launch app for the nearest image from the click.
def display_RenameApp(event, ax):
    ix, iy = event.xdata, event.ydata
    dist = distance_matrix(result, np.array([ix, iy], ndmin=2))
    index = np.argmin(dist)
    if (np.min(dist) < 100):
        app = RenameApp(init_name=names_list[index], index=index)
        app.launch()


fig, ax = scatter_image_plot(result, mini_img)
click = Click(ax, display_RenameApp, button=1)

plt.get_current_fig_manager().window.maxsize()
plt.show()



