import numpy as np
import pandas as pd
from Analytics import Smooth, KL_divergence, CalculateReactionLevel, ConfusionPlot, PlotIntensity, ShowFrame
import cv2
# show a particular frame/moment of video
video_path = "examples/Shan.mp4"
ShowFrame(video_path, frame = 10, save_loc = 'ExampleAnalysis/image_frame.png')
ShowFrame(video_path, time = 5, save_loc = 'ExampleAnalysis/image_time.png') #time measured in seconds

# show a happiness curve 
df = pd.read_csv('examples/Shan.csv', index_col=False)
video = cv2.VideoCapture(video_path) # read video
fps = video.get(cv2.CAP_PROP_FPS) #find fps of the video
time_stamp = df['Timestamp'] # get timestamp colomn from dataframe
window = 20
emotions = ['Happy', 'Neutral']
time_series = [Smooth(df[emotion], window = window) for emotion in emotions] #a list of two time series reflecting happiness and neutralness
title = 'Happiness and Neutralness'
PlotIntensity(time_stamp[:-window], time_series, title = title, labels = emotions, save_loc = 'ExampleAnalysis/' + title + '.png')

# calculate reaction curve
reaction = CalculateReactionLevel(df, smooth = True)
PlotIntensity(time_stamp[:-window], [reaction], save_loc = 'ExampleAnalysis/ReactionCurve.png', labels = ['reaction'], title = 'Reaction Curve')

# KL_divergence
df_2 = pd.read_csv('examples/Mark.csv', index_col=False)
kl_divergence = KL_divergence(df, df_2, ['Happy', 'Neutral', 'Sad', 'Angry', 'Surprise'])
print('KL-divergence between Shan and Mark: %.4f'%(kl_divergence))

#Confusion Plot
people = ['Mark', 'Jingjin', 'Sonja', 'Shan']
emotions = ['Happy', 'Neutral', 'Sad', 'Angry', 'Surprise']
dfs = [pd.read_csv('examples/' + people[i] + '.csv', index_col=False) for i in range(len(people))]

KL = np.zeros([4, 4])
for i in range(len(dfs)):
    for j in range(i, len(dfs)):
        KL[i][j] = KL_divergence(dfs[i], dfs[j], emotions)
        KL[j][i] = KL[i][j]
ConfusionPlot(KL, people, save_loc = 'ExampleAnalysis/ConfusionPlot.png')
    