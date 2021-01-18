import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List

def Smooth(y, window):
    """
    @param           y: smooth the curve by averaging over a sliding window
    @param      window: window for averaging
    @output           : a smoothed version of input curve (as an ndarray)
    """
    out = []
    for i in range(len(y) - window):
        out.append(np.mean(y[i:i+window]))
    return np.asarray(out)
    
def KL_divergence(df1, df2, labels: List):
    """
    @param         df1: pandas dataframe of first person containing emotion time series
    @param         df2: pandas dataframe of second person containing emotion time series
    @param      labels: selected emotions one would like to compute KL-divergence on
    @output           : KL divergence averaged over different emotions
    """
    if len(labels) == 0:
        raise Exception("Please specify labels to include for comparison.")
    diff = 0
    try:
        for label in labels:
            diff += np.mean(abs(df1[label] - df2[label]))
        return diff/len(labels)
    except:
        print("Please enter two valid pandas dataframe, with valid labels included in the dataframe.")
        
def CalculateReactionLevel(df, smooth = True):
    """
    @param     df: datafram of emotions including Neutral
    @param smooth: determines wheter to smooth the curve
    @output      : reaction level (as a numpy array) calculated from non-neutral-ness
    """
    if "Neutral" not in df:
        raise Exception("Please include Neutral in the dataframe")
    if smooth:
        out = 1 - Smooth(df["Neutral"], window = 20)
        df = df[:-20]
        out[df.sum(axis = 1) - df["Timestamp"] < 0.1] = 0 
        return out
    else:
        out = 1 - df["Neutral"]
        out[df.sum(axis = 1) - df["Timestamp"] < 0.1] = 0 
        return out
    
def ConfusionPlot(confusion_plot_2d, labels, save_loc = None):
    """
    @param confusion_plot_2d: a 2-dimensional confusion plot, having same x and y dimension, symmetric along the diagonal.
    @param            labels: name of the people being compared, having same dimension as the input confusion plot.
    @param          save_loc: location to save output plot, if None, will show plot on the screen
    """
    try:
        plt.imshow(confusion_plot_2d, cmap=plt.get_cmap('Blues'))
        plt.xticks(range(len(confusion_plot_2d)), labels)
        plt.yticks(range(len(confusion_plot_2d)), labels)
        plt.colorbar()
        if save_loc:
            plt.savefig(save_loc)
        else:
            plt.show()
    except:
        print("Please enter a 2-dimensional list of lists, the number of labels should be the same as size of confusion plot.")
    
    
def PlotIntensity(time_stamp, ys: List, labels: List, title = None, grid = True, save_loc = None):
    """
    @param time_stamp: time stamps of the time series y.
    @param          y: list of time series of intensity, varying between 0 and 1.
    @param     labels: list of emotions corresponding to each time series in ys.
    @param      title: title of the plot
    @param       grid: toggle grid in the plot
    @param   save_loc: location to save output plot, if None, will show plot on the screen
    """
    
    if len(ys) != len(labels):
        raise Exception("Please ensure the number of time series to plot the same as the number of labels")
        
    plt.figure(figsize = [20, 3])
    for i in range(len(ys)):
        plt.plot(time_stamp, ys[i], '-', label = labels[i])
    if title:
        plt.title(title)
    if grid:
        plt.grid()
    plt.ylabel('intensity')
    plt.xlabel('second')
    plt.legend(loc = 'upper right')
    plt.ylim([0, 1])
    if save_loc:
        plt.savefig(save_loc)
    else:
        plt.show()

def ShowFrame(video_path, frame = -1, time = -1, save_loc = None):
    """
    @param video_path: path to the target video
    @param      frame: the target frame of the video to show
    @param       time: time stamp of the target frame to show, please only enter one of the two parameters.
    @param   save_loc: location to save output plot, if None, will show plot on the screen
    """
    try:
        video = cv2.VideoCapture(video_path)
    except:
        print("Please enter a valid path to target video")
    
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    
    if frame < 0 and time < 0:
        raise Exception("Please enter a valid frame number or a valid time in seconds")
    if frame > total_frames or int(time * fps) > total_frames:
        raise Exception("Requested frame out of range")
    
    if frame > 0:
        for i in range(frame):
            ok, image = video.read()
    else:
        frame = int(time * fps)
        for i in range(frame):
            ok, image = video.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    if save_loc:
        plt.savefig(save_loc)
    else:
        plt.show()