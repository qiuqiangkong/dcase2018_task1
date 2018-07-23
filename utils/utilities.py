import numpy as np
import soundfile
import librosa
import os
from sklearn import metrics
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import config


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):

    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '%04d.log' % i1)):
        i1 += 1

    log_path = os.path.join(log_dir, '%04d.log' % i1)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


def read_audio(path, target_fs=None):

    (audio, fs) = soundfile.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs


def calculate_scalar(x):

    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def scale(x, mean, std):

    return (x - mean) / std


def inverse_scale(x, mean, std):

    return x * std + mean


def calculate_accuracy(target, predict, classes_num, average=None):
    """Calculate accuracy.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)

    Outputs:
      accuracy: float
    """

    samples_num = len(target)
    
    correctness = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(samples_num):
        
        total[target[n]] += 1
        
        if target[n] == predict[n]:
            correctness[target[n]] += 1

    accuracy = correctness / total

    if average is None:
        return accuracy
        
    elif average == 'macro':
        return np.mean(accuracy)
        
    else:
        raise Exception('Incorrect average!')


def calculate_confusion_matrix(target, predict, classes_num):
    """Calculate confusion matrix.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
      classes_num: int, number of classes

    Outputs:
      confusion_matrix: (classes_num, classes_num)
    """

    confusion_matrix = np.zeros((classes_num, classes_num))
    samples_num = len(target)

    for n in range(samples_num):
        confusion_matrix[target[n], predict[n]] += 1

    return confusion_matrix


def print_accuracy(class_wise_accuracy, labels):

    print('{:<30}{}'.format('Scene label', 'accuracy'))
    print('------------------------------------------------')
    for (n, label) in enumerate(labels):
        print('{:<30}{:.3f}'.format(label, class_wise_accuracy[n]))
    print('------------------------------------------------')
    print('{:<30}{:.3f}'.format('Average', np.mean(class_wise_accuracy)))


def plot_confusion_matrix(confusion_matrix, title, labels, values):
    """Plot confusion matrix.

    Inputs:
      confusion_matrix: matrix, (classes_num, classes_num)
      labels: list of labels
      values: list of values to be shown in diagonal

    Ouputs:
      None
    """

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

    if labels:
        ax.set_xticklabels([''] + labels, rotation=90, ha='left')
        ax.set_yticklabels([''] + labels)
        ax.xaxis.set_ticks_position('bottom')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    for n in range(len(values)):
        plt.text(n - 0.4, n, '{:.2f}'.format(values[n]), color='yellow')

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.tight_layout()
    plt.show()


def write_leaderboard_submission(submission_path, audio_names, predictions):
    
    ix_to_lb = config.ix_to_lb
    
    f = open(submission_path, 'w')	
    f.write('Id,Scene_label\n')
    
    for n in range(len(audio_names)):
        f.write('{}'.format(os.path.splitext(audio_names[n])[0]))
        f.write(',')
        f.write(ix_to_lb[predictions[n]])
        f.write('\n')
        
    f.close()
    
    logging.info('Write result to {}'.format(submission_path))
    
     
def write_evaluation_submission(submission_path, audio_names, predictions):
    
    ix_to_lb = config.ix_to_lb
    
    f = open(submission_path, 'w')	
    
    for n in range(len(audio_names)):
        f.write('audio/{}'.format(audio_names[n]))
        f.write('\t')
        f.write(ix_to_lb[predictions[n]])
        f.write('\n')
        
    f.close()
    
    logging.info('Write result to {}'.format(submission_path))