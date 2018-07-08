import os
import numpy as np
import argparse
import h5py
import math
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data_generator import DataGenerator, TestDataGenerator
from utilities import (create_folder, get_filename, create_logging,
                       move_data_to_gpu, calculate_confusion_matrix,
                       calculate_accuracy, plot_confusion_matrix)
from models_pytorch import BaselineCnn
import config


def evaluate(model, generator, data_type, devices, max_iteration, cuda):
    """
    model: object.
    generator: object.
    data_type: 'train' | 'validate'.
    max_iteration: int.
    cuda: bool.
    """
    '''
    (output, target) = forward(model=model,
                               generator=generator,
                               data_type=data_type,
                               devices=devices,
                               max_iteration=-1,
                               cuda=cuda)
    '''
    
    generate_func = generator.generate_validate(
            data_type=data_type, devices=devices, max_iteration=max_iteration)
    (output, target) = forward(model=model, generate_func=generate_func, cuda=cuda)

    predict = np.argmax(output, axis=-1)

    classes_num = output.shape[-1]
    confusion_matrix = calculate_confusion_matrix(target, predict, classes_num)
    accuracy = calculate_accuracy(target, predict)

    return accuracy

def forward(model, generate_func, cuda):
    """Forward data to a model.
    """
    

    model.eval()

    output_all = []
    target_all = []

    iteration = 0

    # Evaluate on mini-batch
    for (batch_x, batch_y) in generate_func:
        
        # import crash
        # asdf

        batch_x = move_data_to_gpu(batch_x, cuda)

        batch_output = model(batch_x)

        output_all.append(batch_output.data.cpu().numpy())
        target_all.append(batch_y)

        iteration += 1

    output_all = np.concatenate(output_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)

    return output_all, target_all

'''
def forward(model, generator, data_type, devices, max_iteration, cuda):
    """Forward data to a model.

    model: object.
    generator: object.
    data_type: 'train' | 'validate'.
    max_iteration: int.
    cuda: bool.
    """

    model.eval()

    output_all = []
    target_all = []

    iteration = 0

    # Evaluate on mini-batch
    for (batch_x, batch_y) in generator.generate_validate(
            data_type=data_type, devices=devices, max_iteration=max_iteration):

        batch_x = move_data_to_gpu(batch_x, cuda)

        batch_output = model(batch_x)

        output_all.append(batch_output.data.cpu().numpy())
        target_all.append(batch_y)

        iteration += 1

    output_all = np.concatenate(output_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)

    return output_all, target_all
'''


def train(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    filename = args.filename
    validate = args.validate
    mini_data = args.mini_data
    cuda = args.cuda

    labels = config.labels

    if 'mobile' in subdir:
        devices = ['a', 'b', 'c']
    else:
        devices = ['a']

    batch_size = 64
    classes_num = len(labels)

    # Paths
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 'mini_development.h5')
    else:
        hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                                 'development.h5')

    if validate:
        
        dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                    'fold1_train.txt')
                                    
        dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                        'fold1_evaluate.txt')
                                        
    else:
        dev_train_csv = None
        dev_validate_csv = None

    models_dir = os.path.join(workspace, 'models', subdir, filename,
                              'validate={}'.format(validate))

    create_folder(models_dir)

    # Model
    model = BaselineCnn(classes_num)

    if cuda:
        model.cuda()

    # Data generator
    generator = DataGenerator(hdf5_path=hdf5_path,
                        batch_size=batch_size,
                        dev_train_csv=dev_train_csv,
                        dev_validate_csv=dev_validate_csv)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0.)

    iteration = 0
    train_bgn_time = time.time()

    # Train on mini batches
    for (batch_x, batch_y) in generator.generate_train():

        # Evaluate
        if iteration % 100 == 0:

            train_fin_time = time.time()

            tr_acc = evaluate(model=model,
                              generator=generator,
                              data_type='train',
                              devices=devices,
                              max_iteration=-1,
                              cuda=cuda)

            logging.info("tr_acc: {:.3f}".format(tr_acc))

            if validate:
                
                va_acc = evaluate(model=model,
                                generator=generator,
                                data_type='validate',
                                devices=devices,
                                max_iteration=-1,
                                cuda=cuda)
                                
                logging.info("va_acc: {:.3f}".format(va_acc))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                "iteration: {}, train time: {:.3f} s, validate time: {:.3f} s".format(
                    iteration, train_time, validate_time))

            logging.info("------------------------------------")

            train_bgn_time = time.time()

        # Move data to gpu
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        # Train
        model.train()
        output = model(batch_x)
        loss = F.nll_loss(output, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1

        # Save model
        if iteration % 1000 == 0 and iteration > 0:

            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }
            save_out_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info("Model saved to {}".format(save_out_path))


def inference_validation(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda

    labels = config.labels

    if 'mobile' in subdir:
        devices = ['a', 'b', 'c']
    else:
        devices = ['a']

    validation = True
    batch_size = 64
    classes_num = len(labels)

    # Paths
    hdf5_path = os.path.join(workspace, 'features', 'logmel', subdir,
                             'development.h5')

    dev_train_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                 'fold1_train.txt')
                                 
    dev_validate_csv = os.path.join(dataset_dir, subdir, 'evaluation_setup',
                                    'fold1_evaluate.txt')

    model_path = os.path.join(workspace, 'models', subdir, filename,
                              'validation={}'.format(validation),
                              'md_{}_iters.tar'.format(iteration))

    # Load model
    model = BaselineCnn(classes_num)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Predict & evaluate
    for device in devices:

        print("Device: {}".format(device))

        # Data generator
        generator = DataGenerator(hdf5_path=hdf5_path,
                                  batch_size=batch_size,
                                  dev_train_csv=dev_train_csv,
                                  dev_validate_csv=dev_validate_csv)

        (output, target) = forward(model=model,
                                   generator=generator,
                                   data_type='validate',
                                   devices=device,
                                   max_iteration=-1,
                                   cuda=cuda)

        predict = np.argmax(output, axis=-1)

        classes_num = output.shape[-1]
        
        # Evaluate
        confusion_matrix = calculate_confusion_matrix(
            target, predict, classes_num)
            
        accuracy = calculate_accuracy(target, predict)
        
        class_wise_acc = np.diag(confusion_matrix) / \
            np.sum(confusion_matrix, axis=0)

        print("confusion_matrix: \n", confusion_matrix)
        print("averaged accuracy: {}".format(accuracy))

        # Plot confusion matrix
        plot_confusion_matrix(
            confusion_matrix,
            title='Device {}'.format(device.upper()), 
            labels=labels,
            values=class_wise_acc)
            
            
def inference_testing_data(args):
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    dev_subdir = args.dev_subdir
    test_subdir = args.test_subdir
    workspace = args.workspace
    iteration = args.iteration
    filename = args.filename
    cuda = args.cuda

    labels = config.labels

    if 'mobile' in test_subdir:
        devices = ['a', 'b', 'c']
    else:
        devices = ['a']

    batch_size = 64
    classes_num = len(labels)

    # Paths
    dev_hdf5_path = os.path.join(workspace, 'features', 'logmel', dev_subdir,
                             'development.h5')

    test_hdf5_path = os.path.join(workspace, 'features', 'logmel', test_subdir,
                             'test.h5')

    model_path = os.path.join(workspace, 'models', dev_subdir, filename,
                              'validate=False', 
                              'md_{}_iters.tar'.format(iteration))

    # Load model
    model = BaselineCnn(classes_num)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Predict & evaluate
    for device in devices:

        print("Device: {}".format(device))

        # Data generator
        generator = TestDataGenerator(dev_hdf5_path=dev_hdf5_path,
                                      test_hdf5_path=test_hdf5_path, 
                                      batch_size=batch_size)

        generate_func = generator.generate_testing_data()

        output = forward(model=model, generate_func=generate_func, cuda=cuda)

        predict = np.argmax(output, axis=-1)

        classes_num = output.shape[-1]
        
        # Evaluate
        confusion_matrix = calculate_confusion_matrix(
            target, predict, classes_num)
            
        accuracy = calculate_accuracy(target, predict)
        
        class_wise_acc = np.diag(confusion_matrix) / \
            np.sum(confusion_matrix, axis=0)

        print("confusion_matrix: \n", confusion_matrix)
        print("averaged accuracy: {}".format(accuracy))

        # Plot confusion matrix
        plot_confusion_matrix(
            confusion_matrix,
            title='Device {}'.format(device.upper()), 
            labels=labels,
            values=class_wise_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True)
    parser_train.add_argument('--subdir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    
    parser_inference_validation = subparsers.add_parser('inference_validation')
    parser_inference_validation.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation.add_argument('--subdir', type=str, required=True)
    parser_inference_validation.add_argument('--workspace', type=str, required=True)
    parser_inference_validation.add_argument('--iteration', type=int, required=True)
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
                                             
    parser_inference_testing_data = subparsers.add_parser('inference_testing_data')
    parser_inference_testing_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_testing_data.add_argument('--dev_subdir', type=str, required=True)
    parser_inference_testing_data.add_argument('--test_subdir', type=str, required=True)
    parser_inference_testing_data.add_argument('--workspace', type=str, required=True)
    parser_inference_testing_data.add_argument('--iteration', type=int, required=True)
    parser_inference_testing_data.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation':
        inference_validation(args)
        
    elif args.mode == 'inference_testing_data':
        inference_testing_data(args)

    else:
        raise Exception("Error argument!")
