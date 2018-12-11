import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math as mt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import tkinter as tk
from itertools import product

# Custom pick class, one click on a image open a plot with that image.
# But multiple image can be select if the user press shift, the selection finished when shift is released.
class CustomPick():

    def __init__(self, ax, func, file, names_list):

        self.names_list = names_list
        self.file = file

        self.ax = ax
        self.func = func
        self.press = False
        self.ind_set = set()
        self.c1 = self.ax.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.c2 = self.ax.figure.canvas.mpl_connect('key_press_event', self.on_press)
        self.c3 = self.ax.figure.canvas.mpl_connect('key_release_event', self.on_release)


    def on_pick(self, event):
        if self.press is True:
            self.ind_set.add(event.ind[0])
            fig = event.canvas.figure
            self.ax.artists[event.ind[0]].get_children()[1].set_alpha(0.8)
            fig.canvas.draw_idle()
        else:
            self.names_list = self.func({event.ind[0]}, self.file, self.names_list)

    def on_press(self, event):
        if (self.press is False) & (event.key == "shift"):
            self.ind_set.clear()
            self.press = True

    def on_release(self, event):
        if (self.press is True) & (event.key == "shift"):
            fig = event.canvas.figure
            for index in self.ind_set:
                self.ax.artists[index].get_children()[1].set_alpha(0)
            fig.canvas.draw_idle()
            self.press = False
            if len(self.ind_set) > 0:
                self.names_list = self.func(self.ind_set, self.file, self.names_list)

class RenameApp():
    def __init__(self, ind_set, file, names_list):

        self.ind_set = ind_set
        self.ind_set_length = len(ind_set)
        self.file = file
        self.names_list = names_list

        # Declaration of the original image plot
        self.nrows, self.ncols = plot_dimensions(self.ind_set_length)
        self.fig, self.axes = plt.subplots(int(self.nrows), int(self.ncols), squeeze=False)

        # Dict index: init_name, extension, coord on the plot
        self.init_name_dict = {i: [names_list[i].split(".")[0], names_list[i].split(".")[1], coord]
                    for i, coord in zip(list(ind_set), list(product(range(self.nrows), range(self.ncols))))}

        # If only one image is selected, those variables will be useful to avoid repetition
        self.first_ind = list(self.init_name_dict.keys())[0]
        self.first_name = self.init_name_dict[self.first_ind]

        # Filing the plot
        self.display_original_image()

        # Creation of Tkinter window
        self.window = tk.Tk()
        self.entry = tk.Entry(self.window)
        self.entry.insert(10, self.first_name[0])
        self.save_name_button = tk.Button(self.window, text="Change name", command=self.change_name)
        self.restore_init_name_button = tk.Button(self.window, text="Restore init name", command=self.restore_init_name)
        self.quit_button = tk.Button(self.window, text="Quit", command=self.quit)
        self.save_name_button.pack()
        self.restore_init_name_button.pack()
        self.quit_button.pack()
        self.entry.pack()

        # Display plot and Tkinter window
        plt.show(self.fig)
        self.window.mainloop()

    def display_original_image(self):
        [axi.set_axis_off() for axi in self.axes.ravel()]
        for ind in self.init_name_dict.keys():
            x, y = self.init_name_dict[ind][2]
            self.axes[x, y].imshow(mpimg.imread(self.file + self.names_list[ind]))
            self.axes[x, y].set_title(self.init_name_dict[ind][0])

    # When change_name or restore_name are called, the plot need to update the titles of each axes
    def update_plot(self):
        for ind in self.init_name_dict.keys():
            x, y = self.init_name_dict[ind][2]
            self.axes[x, y].set_title(self.names_list[ind].split(".")[0])
            self.fig.canvas.draw_idle()

    def change_name(self):
        if len(self.entry.get().split(".")) < 2:
            user_entry = self.entry.get()
            if len(self.ind_set) == 1:
                new_name = user_entry + "." + self.first_name[1]
                os.rename(self.file + self.names_list[self.first_ind], self.file + new_name)
                self.names_list[self.first_ind] = new_name
                self.fig.canvas.set_window_title(user_entry)
            else:
                n = 1
                for ind in self.ind_set:
                    new_name = self.entry.get() + "_" + str(n) + "." + self.init_name_dict[ind][1]
                    os.rename(self.file + self.names_list[ind], self.file + new_name)
                    self.names_list[ind] = new_name
                    n += 1
            self.update_plot()
        else:
            print("Error no point allowed in the entry")

    def restore_init_name(self):
        for ind in self.ind_set:
            init_name = self.init_name_dict[ind][0] + "." + self.init_name_dict[ind][1]
            os.rename(self.file + self.names_list[ind], self.file + init_name)
            self.names_list[ind] = init_name

        if len(self.ind_set) == 1:
            self.fig.canvas.set_window_title(self.names_list[self.first_ind].split(".")[0])
            self.entry.delete(first=0, last=22)
            self.entry.insert(10, self.names_list[self.first_ind].split(".")[0])

        self.update_plot()

    def quit(self):
        self.window.quit()
        self.window.destroy()
        plt.close(self.fig)


# Scatter image plot
def image_plot(X_2D, images, figsize, image_zoom=1):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(X_2D[:, 0], X_2D[:, 1], 'o', picker=25)
    plt.axis('off')
    artists = []
    for xy, i in zip(X_2D, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=True, bboxprops=dict(facecolor='blue', edgecolor="white", alpha=0))
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2D)
    ax.autoscale()
    return fig, ax

def plot_dimensions(length):
    nrows = mt.ceil(mt.sqrt(length))
    ncols = length // nrows
    if length % nrows != 0:
        ncols += 1
    return int(nrows), int(ncols)

#Scatter plot
def classic_plot(X_2D):
    fig, ax = plt.subplots()
    ax.plot(X_2D[:, 0], X_2D[:, 1],'o', picker=5)
    return fig, ax

####################################################
# Function to pass trought CustomPick

# Display an adaptable plot to show images
def display_original_img(ind_set, file, names_list):
    fig = plt.figure()
    nrows = mt.ceil(mt.sqrt(len(ind_set)))
    ncols = len(ind_set) // nrows
    if len(ind_set) % nrows != 0:
        ncols = ncols+1
    n = 1
    for ind in ind_set:
        plt.subplot(nrows, ncols, n)
        n += 1
        plt.axis('off')
        plt.imshow(mpimg.imread(file + names_list[ind]))
    plt.show()
    return fig

def display_RenameApp(ind_set, file, names_list):
    app = RenameApp(ind_set, file, names_list)
    return app.names_list