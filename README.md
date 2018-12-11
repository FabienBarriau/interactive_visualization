# interactive_visualization
Show your natural image in a 2D space thanks to CNN representation and dimensional reduction method.

# Running 

python interactive_visualization.py --file image/ --visu TSNE --model VGG16

Don't forget the "/" at the end of your file.
You've got the choice between "PCA" and "TSNE" for dimensional reduction method.
You've got the choice between "VGG16" and "VGG19" for features extracting method.

![Alt](/Example/fish_general_view.png "example with different species of fish")
![Alt](/Example/natural_landscape_object_general_view.png "example with landscape and some objects")

# Renamer App

When you click on a mini image, the original image open in a little window that allow you to change the name's file. That's a new way to rename your photos !

![Alt](/Example/fish_zoom_and_rename.png "renamer app")

multi selection

![Alt](/Example/fish_selection "selection")

multi rename

![Alt](/Example/fish_multi_rename "multi rename")

# Requirements

Python 3.5
* Tensorflow
* Scipy
* Pillow
* Tkinter
* Numpy
* Matplotlib
* (argparse, os)
