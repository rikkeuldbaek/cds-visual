#function for making histogram of an object

import os
import cv2
import matplotlib.pyplot as plt



### making a function that takes a filename
def col_hist_func(filename):
    # set data path
    in_path = os.path.join("..", "data", "img", filename)

    #loading the file
    image = cv2.imread(in_path)

    #split image
    channels = cv2.split(image)
    # names of colours
    colors = ("b", "g", "r")
    # create plot
    plt.figure()
    # add title
    plt.title("Histogram")
    # Add xlabel
    plt.xlabel("Bins")
    # Add ylabel
    plt.ylabel("# of Pixels")

    # for every tuple of channel, colour
    for (channel, color) in zip(channels, colors): #zip two iterable objects 3*3 of same length = into 3*3 elements in a list
        # Create a histogram
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        # Plot histogram
        plt.plot(hist, color=color)
        # Set limits of x-axis
        plt.xlim([0, 256])
    #write the hist

    #write outpath
    hist_filename = f'hist_{filename}'
    out_path = os.path.join("..", "out", hist_filename) #make f string here
    plt.savefig(out_path)
    # Show plot
    plt.show()




#out_path = os.path.join("..", "data", "img", "trex_hist.png")
#col_hist_func("trex.png", out_path)