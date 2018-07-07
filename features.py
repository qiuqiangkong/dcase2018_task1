import numpy as np
import os
import pandas as pd
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import csv
import random

from utilities import read_audio, create_folder
import config


class LogMelExtractor():
    def __init__(self, sample_rate, window_size, overlap, mel_bins):
        
        self.window_size = window_size
        self.overlap = overlap
        self.ham_win = np.hamming(window_size)
        
        self.melW = librosa.filters.mel(sr=sample_rate, 
                                        n_fft=window_size, 
                                        n_mels=mel_bins, 
                                        fmin=20., 
                                        fmax=sample_rate // 2).T
    
    def transform(self, audio):
    
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap
    
        [f, t, x] = signal.spectral.spectrogram(
                        audio, 
                        window=ham_win,
                        nperseg=window_size, 
                        noverlap=overlap, 
                        detrend=False, 
                        return_onesided=True, 
                        mode='magnitude') 
        x = x.T
            
        x = np.dot(x, self.melW)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32)
        
        return x


def calculate_logmel(audio_path, sample_rate, feature_extractor):
    
    # Read audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate)
    
    '''We do not divide the maximum value of an audio here because we assume 
    the low energy of an audio may also contain information of a scene. '''
    
    # Extract feature
    feature = feature_extractor.transform(audio)
    
    return feature


def read_development_meta(meta_csv):
    
    df = pd.read_csv(meta_csv, sep='\t')
    df = pd.DataFrame(df)
    
    audio_names = []
    scene_labels = []
    identifiers = []
    source_labels = []
    
    for row in df.iterrows():
        
        audio_name = row[1]['filename'].split('/')[1]
        scene_label = row[1]['scene_label']
        identifier = row[1]['identifier']
        source_label = row[1]['source_label']
        
        audio_names.append(audio_name)
        scene_labels.append(scene_label)
        identifiers.append(identifier)
        source_labels.append(source_label)
        
    return audio_names, scene_labels, identifiers, source_labels
    
    
def read_leaderboard_meta(evaluation_csv):
    
    with open(evaluation_csv, 'r') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    audio_names = []
        
    for li in lis:
        audio_name = li[0].split('/')[1]
        audio_names.append(audio_name)
        
    return audio_names
    

def calculate_features(args):
    
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    data_type = args.data_type
    workspace = args.workspace
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    
    # Paths
    audio_dir = os.path.join(dataset_dir, subdir, 'audio')
    
    if data_type == 'development':
        meta_csv = os.path.join(dataset_dir, subdir, 'meta.csv')
    else:
        evaluation_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup', 'test.txt')
    
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir, 
                                 'mini_data.h5')
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir, 'data.h5')
        
        
    create_folder(os.path.dirname(hdf5_path))
    
    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)

    # Read meta csv
    if data_type == 'development':
        [audio_names, scene_labels, identifiers, source_labels] = read_development_meta(meta_csv)
        
    elif data_type == 'leaderboard':
        audio_names = read_leaderboard_meta(evaluation_csv)

    # Only use partial data when set mini_data to True
    if mini_data:
        
        audios_num = 300
        random_state = np.random.RandomState(0)
        audio_indexes = np.arange(len(audio_names))
        random_state.shuffle(audio_indexes)
        audio_indexes = audio_indexes[0 : audios_num]
        
        audio_names = [audio_names[idx] for idx in audio_indexes]
        
        if data_type == 'development':
            scene_labels = [scene_labels[idx] for idx in audio_indexes]
            identifiers = [identifiers[idx] for idx in audio_indexes]
            source_labels = [source_labels[idx] for idx in audio_indexes]
        
    print("Number of audios: {}".format(len(audio_names)))
    
    # Create hdf5 file
    hf = h5py.File(hdf5_path, 'w')
    
    hf.create_dataset(
        name='feature', 
        shape=(0, seq_len, mel_bins), 
        maxshape=(None, seq_len, mel_bins), 
        dtype=np.float32)
    
    n = 0
    calculate_time = time.time()
    
    for audio_name in audio_names:
        
        print(n, audio_name)
        
        # Calculate feature
        audio_path = os.path.join(audio_dir, audio_name)
        
        # Extract feature
        feature = calculate_logmel(audio_path=audio_path, 
                                    sample_rate=sample_rate, 
                                    feature_extractor=feature_extractor)
        '''(seq_len, mel_bins)'''
        
        print(feature.shape)
        
        hf['feature'].resize((n + 1, seq_len, mel_bins))
        hf['feature'][n] = feature
        
        # Plot log Mel for debug
        if False:
            plt.matshow(feature.T, origin='lower', aspect='auto', cmap='jet')
            plt.show()
            
        n += 1
        
    # Write meta info to hdf5
    hf.create_dataset(name='filename', data=[s.encode() for s in audio_names], dtype='S50')
    
    if data_type == 'development':
        hf.create_dataset(name='scene_label', data=[s.encode() for s in scene_labels], dtype='S20')
        hf.create_dataset(name='identifier', data=[s.encode() for s in identifiers], dtype='S20')
        hf.create_dataset(name='source_label', data=[s.encode() for s in source_labels], dtype='S20')

    hf.close()
    
    print("Write out hdf5 file to {}".format(hdf5_path))
    print("Time spent: {} s".format(time.time() - calculate_time))


def logmel(args):

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    data_type = args.data_type
    workspace = args.workspace
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    
    # Paths
    audio_dir = os.path.join(dataset_dir, subdir, 'audio')
    
    meta_csv = os.path.join(dataset_dir, subdir, 'meta.csv')
    
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir, 
                                 'mini_data.h5')
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir, 'data.h5')
        
        
    create_folder(os.path.dirname(hdf5_path))
    
    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)

    audio_names = []
    scene_labels = []
    identifiers = []
    source_labels = []
    
    # Read meta csv file
    df = pd.read_csv(meta_csv, sep='\t')
    df = pd.DataFrame(df)
    
    for row in df.iterrows():
        
        audio_name = row[1]['filename'].split('/')[1]
        scene_label = row[1]['scene_label']
        identifier = row[1]['identifier']
        source_label = row[1]['source_label']
        
        audio_names.append(audio_name)
        scene_labels.append(scene_label)
        identifiers.append(identifier)
        source_labels.append(source_label)
        
    if mini_data:
        
        random_state = np.random.RandomState(0)
        audios_num = 3
        audio_indexes = np.arange(audios_num)
        random_state.shuffle(audio_indexes)
        
        audio_names = [audio_names[idx] for idx in audio_indexes]
        scene_labels = [scene_labels[idx] for idx in audio_indexes]
        identifiers = [identifiers[idx] for idx in audio_indexes]
        source_labels = [source_labels[idx] for idx in audio_indexes]
        
    print("Number of audios: {}".format(len(audio_names)))
    
    hf = h5py.File(hdf5_path, 'w')
    
    hf.create_dataset(
        name='feature', 
        shape=(0, seq_len, mel_bins), 
        maxshape=(None, seq_len, mel_bins), 
        dtype=np.float32)
    
    n = 0
    
    for audio_name in audio_names:
        
        print(audio_name)
        
        # Calculate feature
        audio_path = os.path.join(audio_dir, audio_name)
        
        # Extract feature
        feature = calculate_logmel(audio_path=audio_path, 
                                    sample_rate=sample_rate, 
                                    feature_extractor=feature_extractor)
        '''(seq_len, mel_bins)'''
        
        print(feature.shape)
        
        hf['feature'].resize((n + 1, seq_len, mel_bins))
        hf['feature'][n] = feature
        
        # Plot log Mel for debug
        if False:
            plt.matshow(feature.T, origin='lower', aspect='auto', cmap='jet')
            plt.show()
            
        n += 1

    hf.close()

"""
def logmel(args):

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    data_type = args.data_type
    workspace = args.workspace
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    window_size = config.window_size
    overlap = config.overlap
    seq_len = config.seq_len
    mel_bins = config.mel_bins
    
    # Paths
    audio_dir = os.path.join(dataset_dir, subdir, 'audio')
    
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir, 
                                 'mini_data.h5')
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir, 'data.h5')
        
        
    create_folder(os.path.dirname(hdf5_path))
    
    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate, 
                                        window_size=window_size, 
                                        overlap=overlap, 
                                        mel_bins=mel_bins)
    
    audio_names = os.listdir(audio_dir)
    audio_names = sorted(audio_names)
    
    audios_num = len(audio_names)
    
    # Only calculate features for a small amount of data
    if mini_data:
        
        audios_num = 3
        random.shuffle(audio_names)

    calculate_time = time.time()
    
    # Write out features to hdf5
    with h5py.File(hdf5_path, 'w') as hf:
        
        hf.create_dataset(
            name='feature', 
            shape=(0, seq_len, mel_bins), 
            maxshape=(None, seq_len, mel_bins), 
            dtype=np.float32)
        
        n = 0
        
        for audio_name in audio_names:

            print(n, audio_name)

            # Calculate feature
            audio_path = os.path.join(audio_dir, audio_name)
            
            # Extract feature
            feature = calculate_logmel(audio_path=audio_path, 
                                       sample_rate=sample_rate, 
                                       feature_extractor=feature_extractor)
            
            print(feature.shape)
            
            hf['feature'].resize((n + 1, seq_len, mel_bins))
            hf['feature'][n] = feature
            
            # Plot log Mel for debug
            if False:
                plt.matshow(feature.T, origin='lower', aspect='auto', cmap='jet')
                plt.show()
                
            n += 1
            
            if n == audios_num:
                break

        filenames = [audio_name.encode() for audio_name in audio_names[0 : audios_num]]
        hf.create_dataset(name='filename', data=filenames, dtype='S50')
                   
    print("Write out to {}".format(hdf5_path))
    print("Time: {} s".format(time.time() - calculate_time))
    
    # If development data then append label informations to hdf5
    if data_type == 'development':
        
        meta_csv = os.path.join(dataset_dir, subdir, 'meta.csv')
        
        df = pd.read_csv(meta_csv, sep='\t')
        df = pd.DataFrame(df)
        
        with h5py.File(hdf5_path, 'r+') as hf:
            
            
        
        append_meta_to_hdf5(hdf5_path, metas_dir)
        
        check_hdf5(hdf5_path)
"""

def append_meta_to_hdf5(hdf5_path, metas_dir):
    
    train_meta_csv = os.path.join(metas_dir, 'fold1_train.txt')
    test_meta_csv = os.path.join(metas_dir, 'fold1_test.txt')
    
    with h5py.File(hdf5_path, 'r+') as hf:
            
        audio_names = hf['filename'][:]
        
        audios_num = len(audio_names)
        
        hf.create_dataset(name='label', shape=(audios_num,), dtype='S20')
        hf.create_dataset(name='location', shape=(audios_num,), dtype='S20')
        # hf.create_dataset(name='validation', shape=(audios_num,), dtype=np.int32)
                    
        # update_hdf5(hf, train_meta_csv, validation=0)
        # update_hdf5(hf, test_meta_csv, validation=1)

        print("Write meta data to {}".format(hdf5_path))

def update_hdf5(hf, csv_file):
    
    df = pd.read_csv(csv_file, sep='\t')
    df = pd.DataFrame(df)
    
    

'''
def update_hdf5(hf, csv_file, validation):
    
    with open(csv_file, 'r') as csv_f:
        reader = csv.reader(csv_f, delimiter='\t')
        lis = list(reader)
        
        for li in lis:
            name = li[0].split('/')[1]
            label = name.split('-')[0]
            location = name.split('-')[1]
            indexes = np.where(hf['audio_name'][:] == name.encode())[0]
            
            if len(indexes) > 0:
                index = indexes[0]

                hf['label'][index] = label.encode()
                hf['location'][index] = location.encode()
                # hf['validation'][index] = validation
'''
            

def check_hdf5(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        labels = hf['label'][:]
        for label in labels:
            if len(label) == 0:
                raise Exception("Some audio does not exist in meta file!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_logmel = subparsers.add_parser('logmel')
    parser_logmel.add_argument('--dataset_dir', type=str, required=True)
    parser_logmel.add_argument('--subdir', type=str, required=True)
    parser_logmel.add_argument('--data_type', type=str, required=True, choices=['development', 'leaderboard'])
    parser_logmel.add_argument('--workspace', type=str, required=True)
    parser_logmel.add_argument('--mini_data', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if args.mode == 'logmel':
        
        calculate_features(args)
        
        # if args.data_type == 'development':
        #     calculate_features(args)
        #     
        # elif args.data_type == 'leaderboard':
        #     calculate_features(args)
        
    else:
        raise Exception("Incorrect arguments!")
        