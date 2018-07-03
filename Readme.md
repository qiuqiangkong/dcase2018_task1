# DCASE 2018 Task 1 Acoustic Scene Classification

DCASE 2018 Task 1 acoustic scene classification is a challenge to classifiy a 10 second audio clip to one of 10 classes such as 'airport', 'urban park', etc. We provide a convolutional neural network (CNN) baseline system implemented with PyTorch in this code base. More details about this challenge can be found http://dcase.community/challenge2018/task-acoustic-scene-classification

### DATASET

The dataset is downloadable from http://dcase.community/challenge2018/task-acoustic-scene-classification

The dataset contains 10 classes of audio scenes, recorded with Device A, B and C. The statistic of the data is shown below:

|           |         Attributes        |                    Dev.                    | Test |
|:---------:|:-------------------------:|:------------------------------------------:|:----:|
| Subtask A | Binanural, 48 kHz, 24 bit |               Device A: 8640               | 1200 |
| Subtask B |       Mono, 44.1 kHz      | Device A: 8640 Device B: 720 Device C: 720 | 2400 |
| Subtask C |             -             |                  Any data                  | 1200 |

The log mel spectrogram of the scenes are shown below:

![alt text](appendix/logmel.png)


### Result



### Extra link

The official baseline system implemented using Keras can be found https://github.com/DCASE-REPO/dcase2018_baseline
