"""
EmotionCSV.py: Script for emotion recognition for a pre-recorded video, produce a time series of emotions

Usage:
    EmotionCSV.py -i <file> -o <file> -m <file> [options]

Options:
    -h --help                               show this screen
    -i <file>, --input <file>               input recorded video
    -o <file>, --output <file>              output csv file
    -m <file>, --model <file>               emotion recognition model to use (image classification)
"""

from docopt import docopt
import cv2
import matplotlib.pyplot as plt
import time
import tensorflow as tf 
from scipy import ndimage
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def detectFace(frame, detector):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    
    f = detector.detectMultiScale(frame_gray)
    if len(f) > 0:
        (x, y, w, h) = f[0]
        xe = min(frame_gray.shape[0], x + w * 1.1)
        ye = min(frame_gray.shape[1], y + h * 1.1)
        x = max(0, x - w * 0.1)
        y = max(0, y - h * 0.1)
        start_point = (int(round(x)), int(round(y)))
        end_point = (int(round(xe)), int(round(ye)))
        #frame = cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 4)
        box = (x, y, w * 1.2, h * 1.2)
    else:
        box = None
    return box
    
def FaceTracker(model, video, detector, output_file, total_frames):
    print('Timestamp,Neutral,Happy,Sad,Surprise,Angry', file = output_file)
    count = 0
    while True:
        print('Processing: [%d/%d] frames'%(count, total_frames), end = '\r')
        ok, frame = video.read()
        if not ok:
            break
        if count % 10 == 0:
            tracker = cv2.TrackerKCF_create() 
            box = detectFace(frame, detector)
            if not box:
                print(count, end = ',', file = output_file)
                for i in range(5):
                    print('0.000', end = ',', file = output_file)
                print('', file = output_file)
                count += 1
                continue
            box = tuple(box)
            ok = tracker.init(frame, box)
            rec_frame = frame[int(box[1]):int(box[1] + box[3]), int(box[0]): int(box[0] + box[2])]
            rec_res = ExpressionRecognition(model, rec_frame)
            if rec_res != 'Unsuccessful':
                print(count, end = ',', file = output_file)
                for i in range(5):
                    print('%.3f'%rec_res[0][i], end = ',', file = output_file)
                print('', file = output_file)
        else:
            ok, box = tracker.update(frame)
            if ok:
                (x, y, w, h) = box
                x, y, w, h = int(x), int(y), int(w), int(h)
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4, 1)

                size = max(w, h)
                mid = (x + w * 0.5, y + h * 0.5)
                rec_frame = frame[int(mid[1] - size * 0.5):int(mid[1] + size * 0.5),  int(mid[0] - size * 0.5):int(mid[0] + size * 0.5)]
                rec_res = ExpressionRecognition(model, rec_frame)
                if rec_res != 'Unsuccessful':
                    print(count, end = ',', file = output_file)
                    for i in range(5):
                        print('%.3f'%rec_res[0][i], end = ',', file = output_file)
                    print('', file = output_file)
                else:
                    print(count, end = ',', file = output_file)
                    for i in range(5):
                        print('0.000', end = ',', file = output_file)
                    print('', file = output_file)
            else:
                print(count, end = ',', file = output_file)
                for i in range(5):
                    print('0.000', end = ',', file = output_file)
                print('', file = output_file)
        #cv2.imshow("Tracking", frame)
        count += 1
        
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
            
def ExpressionRecognition(model, frame):
    #input size for this particular model is 48 * 48
    if frame.shape[0] < 48 or frame.shape[1] < 48:
        return 'Unsuccessful'
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = ndimage.zoom(frame, 48/max(frame.shape))
        if min(frame.shape) < 48:
            frame = np.pad(frame, ((48 - frame.shape[0],0), (48 - frame.shape[1],0)), 'edge')
        frame = np.expand_dims(frame, axis = 0)
        frame = np.expand_dims(frame, axis = -1)
        frame = frame/255.0
        return model.predict(frame)

def main():
    args = docopt(__doc__)
    model_path = args['--model']
    video_path = args['--input']
    output_file = open(args['--output'], 'w+')
    model = tf.keras.models.load_model(model_path, compile = False)

    emotion_labels = {0:'Neutral',1:'Happy',2:'Sad',3:'Surprise',
                      4:'Angry',5:'Unsuccessful'}
    emotions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Angry']
    detector =  cv2.CascadeClassifier()
    detector.load(cv2.samples.findFile('FaceTracker/haarcascade_frontalface_default.xml'))
    video = cv2.VideoCapture(video_path)

    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    FaceTracker(model, video, detector, output_file, total_frames)
    output_file.close()
    print('Video Processing Finished!')
    
if __name__ == '__main__':
    main()
