import numpy as np
import h5py
import csv
import time
import logging

from utilities import calculate_scalar, scale
import config


class DataGenerator(object):

    def __init__(self, hdf5_path, batch_size, dev_train_csv=None,
                 dev_validate_csv=None, seed=1234):
        """
        Inputs:
          hdf5_path: string
          batch_size: int
          dev_train_csv: string | None, if None then use all data for training
          dev_validate_csv: string | None, if None then use all data for training
          seed: int, random seed
        """

        self.batch_size = batch_size

        self.random_state = np.random.RandomState(seed)
        lb_to_ix = config.lb_to_ix

        # Load data
        load_time = time.time()
        hf = h5py.File(hdf5_path, 'r')

        self.x = hf['feature'][:]
        self.audio_names = [s.decode() for s in hf['filename'][:]]
        self.scene_labels = [s.decode() for s in hf['scene_label'][:]]
        self.identifiers = [s.decode() for s in hf['identifier'][:]]
        self.source_labels = [s.decode() for s in hf['source_label']]
        self.y = np.array([lb_to_ix[lb] for lb in self.scene_labels])

        hf.close()
        logging.info("Loading data time: {}".format(time.time() - load_time))

        # Calculate scalar
        (self.mean, self.std) = calculate_scalar(self.x)

        # Use all data for training
        if dev_train_csv is None and dev_validate_csv is None:

            self.tr_audio_indexes = np.arange(len(self.audio_names))
            logging.info("Use all development data for training. ")

        # Split data to training and validation
        else:

            self.tr_audio_indexes = self.calculate_audio_indexes_from_csv(
                dev_train_csv)
                
            self.va_audio_indexes = self.calculate_audio_indexes_from_csv(
                dev_validate_csv)
                
            logging.info("Split development data to {} training and {} validation data. ".format(
                len(self.tr_audio_indexes), len(self.va_audio_indexes)))

    def calculate_audio_indexes_from_csv(self, csv_file):
        """Calculate indexes from a csv file. 
        
        Args:
          csv_file: string, path of csv file
        """

        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            lis = list(reader)

        audio_indexes = []

        for li in lis:
            audio_name = li[0].split('/')[1]

            if audio_name in self.audio_names:
                audio_index = self.audio_names.index(audio_name)
                audio_indexes.append(audio_index)

        return audio_indexes

    def generate_train(self):
        """Generate mini-batch data for training. 
        
        Returns:
          batch_x: (batch_size, seq_len, freq_bins)
          batch_y: (batch_size,)
        """

        batch_size = self.batch_size
        audio_indexes = self.tr_audio_indexes
        audios_num = len(audio_indexes)

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:

            # Reset pointer
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            iteration += 1

            batch_x = self.x[batch_audio_indexes]
            batch_y = self.y[batch_audio_indexes]

            # Transform data
            batch_x = self.transform(batch_x)

            yield batch_x, batch_y

    def generate_validate(self, data_type, devices, max_iteration=None):
        """Generate mini-batch data for evaluation. 
        
        Args:
          data_type: 'train' | 'validate'
          devices: list of devices, e.g. ['a'] | ['a', 'b', 'c']
          max_iteration: int, maximum iteration for validation
          
        Returns:
          batch_x: (batch_size, seq_len, freq_bins)
          batch_y: (batch_size,)
        """

        batch_size = self.batch_size

        if data_type == 'train':
            audio_indexes = self.tr_audio_indexes

        elif data_type == 'validate':
            audio_indexes = self.va_audio_indexes

        else:
            raise Exception("Invalid data_type!")

        # Get indexes of specific devices
        devices_specific_indexes = []

        for n in range(len(audio_indexes)):
            if self.source_labels[audio_indexes[n]] in devices:
                devices_specific_indexes.append(audio_indexes[n])

        logging.info("Number of {} audios in specific devices {}: {}".format(
            data_type, devices, len(devices_specific_indexes)))

        audios_num = len(devices_specific_indexes)

        iteration = 0
        pointer = 0

        while True:

            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            # Get batch indexes
            batch_audio_indexes = devices_specific_indexes[
                pointer: pointer + batch_size]
                
            pointer += batch_size

            iteration += 1

            batch_x = self.x[batch_audio_indexes]
            batch_y = self.y[batch_audio_indexes]

            # Transform data
            batch_x = self.transform(batch_x)

            yield batch_x, batch_y

    def transform(self, x):
        """Transform data. 
        
        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)
          
        Returns:
          Transformed data. 
        """

        return scale(x, self.mean, self.std)
