# Emotion Recognition for Recorded Meetings

This is a tool for emotion recognition and emotion time series analysis for recorded meetings.

## Content

Dictories
```
/Models -- contains machine learning models for emotion detection
/FaceTracker --...
```

## Pre-requisite

This tool requires following python packages:

```
opencv-contrib-python
numpy
pandas
matplotlib
scipy
tensorflow
docopt
```

## Get Started

Required packages are installed in conda environment "engagement" on shared Azure Machine. Simply run

```
conda activate engagement
```

To do a emotion recognition, use script `EmotionCSV.py`. An example command line:
```
python3 EmotionCSV.py -i examples/input.mp4 -o examples/output.csv -m Models/expression.model
```

For help, please run 
```
python3 EmotionCSV.py -h
```

Currently the only useful facial expression recoginition model we have access to is Models/expression.model. In the future we should train better models using AffectNet dataset. This version of script also sufferes from relatively slow speed. The input video resolution is best not exceeding 540p. In the future a downsampling module should be implemented to improve computation speed.

EmotionCSV.py produces a csv file containing the probability of each emotion of the face in the video at a given frame, which can be used for further data integration and analysis

## Analytics

With csv file one can do whatever analysis they prefer. For educational and practical purpose, script `Analysis.py` contains several tools for data analysis with produced csv file. These tools includes:

1. Curve smoothing: smooth a time series curve by averaging over a sliding window
2. Calculate KL divergence: caculate KL divergence of two sets of time series (averaged over different emotions)
3. Calculate reaction level: reaction level is defined as non-neutralness.
4. Create Confusion plot: save a confusion plot 
5. Plot a time series: plot one or several timeseries with labels.
6. Save a chosen frame or time stamp in a video, typically used for human verification of machine's prediction.

For detailed documentation of each function please refer to the comment parts of `Analysis.py`

For a demonstration of example product of each analytics function, run 

```
python3 ExampleAnalysis.py
```

And check the result in directory `ExampleAnalysis`.

## Copyright

This work is created by Shan Huang in Oct. 2020. All right reserved.
