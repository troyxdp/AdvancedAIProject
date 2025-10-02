import os
import time
import math
import csv

import numpy as np
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from classes.vae import VariationalAutoencoder
from classes.layer import *
from classes.mnist_dataloader import MnistDataloader

# This class is the same as in my HYP
class EpochStatistics():

    def __init__(self, loss: float, epoch_time: float):
        self._loss = loss
        self._epoch_time = epoch_time

    def get_loss(self):
        return self._loss
    
    def get_epoch_time(self):
        return self._epoch_time

# This code is adapted from my HYP 
def train_model(
        # Dataset paths and info
        train_images_path:str,
        train_labels_path:str,
        val_percentage:float,
        # Output paths and network that is to be trained
        output_folder:str,
        training_network:VariationalAutoencoder,
        # Hyperparameters
        num_epochs:float=120,
        initial_lr:float=0.00005,
        final_lr:float=0.00001,
        initial_kl_beta:float=0.1,
        final_kl_beta:float=1.0,
        kl_beta_step:float=0.1,
        beta_1:float=None,
        beta_2:float=None,
        l2_lambda:float=1e-5,
        clip_score:float=1.0,
        checkpoint_epoch=10,
        early_stop_threshold=20
    ):
        # Get dataset
        print("\nFetching training and validation data...")
        x_train, _ = MnistDataloader(train_images_path, train_labels_path).load_data()

        # Normalize dataset
        print("Normalizing data...")
        for i in range(len(x_train)):
            x_train[i] = np.array(x_train[i])
            x_train[i] = x_train[i] / 255.0

        dataset_size = len(x_train)
        num_val_samples = int(dataset_size * val_percentage)
        num_train_samples = dataset_size - num_val_samples
        print(f"Number of training items: {num_train_samples}")
        print(f"Number of validation items: {num_val_samples}\n")

        # Create output folder (if it does not already exist)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        # lists to store statistics for each epoch
        train_stats = []
        val_stats = []
        best_train_loss = math.inf
        best_val_loss = math.inf
        best_val_loss_epoch = -1

        # start training
        train_start_time = time.time()
        print("Starting training...\n")
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}...")
            # get learning rate
            lr = _determine_epoch_learning_rate(epoch, num_epochs, initial_lr, final_lr)
            kl_beta = _determine_kl_beta(epoch, initial_kl_beta, final_kl_beta, kl_beta_step)

            # train cycle
            train_cycle_start_time = time.time()
            train_error_this_epoch = 0
            for sample in tqdm(x_train[:num_train_samples], desc="Training cycle progress: ", ncols=150):
                # get song and feed it forward through network
                output = training_network.forward(sample)

                # get error of output for statistics
                # Reconstruction loss + KL divergence
                error = (1 / (sample.shape[0] * sample.shape[1])) * np.dot(np.subtract(sample, output).flatten(), np.subtract(sample, output).flatten()) + 0.5 * np.sum(vae.get_mean()**2 + np.exp(vae.get_log_var()) - 1 - vae.get_log_var())
                train_error_this_epoch += error

                # backpropogate
                error_prime = (2 / (sample.shape[0] * sample.shape[1])) * np.subtract(output, sample)
                training_network.back_propogate(lr=lr, error_prime=error_prime, epoch_num=epoch+1, beta_1=beta_1, beta_2=beta_2, kl_beta=kl_beta, l2_lambda=l2_lambda, clip_value=clip_score)

            # save statistics
            train_cycle_time = time.time() - train_cycle_start_time
            train_stat = EpochStatistics(train_error_this_epoch, train_cycle_time)
            train_stats.append(train_stat)
            print(f"Training loss: {float(train_error_this_epoch)}")
            print(f"Average training loss: {float(train_error_this_epoch) / num_train_samples}")

            # validation cycle
            val_error_this_epoch = 0
            val_cycle_start_time = time.time()
            for sample in tqdm(x_train[num_train_samples:], desc="Validation cycle progress: ", ncols=150):
                # get song and feed it forward through network
                sample = np.array(sample)
                output = training_network.forward(sample)

                # get error of output for statistics
                # Reconstruction loss + KL divergence
                error = (1 / (sample.shape[0] * sample.shape[1])) * (np.dot(np.subtract(sample, output).flatten(), np.subtract(sample, output).flatten()) + 0.5 * np.sum(vae.get_mean()**2 + np.exp(vae.get_log_var()) - 1 - vae.get_log_var()))
                val_error_this_epoch += error

            # save statistics 
            val_cycle_time = time.time() - val_cycle_start_time
            val_stat = EpochStatistics(val_error_this_epoch, val_cycle_time)
            val_stats.append(val_stat)
            print(f"Validation loss: {float(val_error_this_epoch)}")
            print(f"Average validation loss: {float(val_error_this_epoch) / num_val_samples}")

            # save model
            if train_error_this_epoch < best_train_loss:
                print("New best train loss!")
                training_network.save_network(os.path.join(output_folder, 'best_train_loss_network.pkl'))
                best_train_loss = train_error_this_epoch
            if val_error_this_epoch < best_val_loss:
                print("New best val loss!")
                training_network.save_network(os.path.join(output_folder, 'best_val_loss_network.pkl'))
                best_val_loss = val_error_this_epoch
                best_val_loss_epoch = epoch
            if epoch % checkpoint_epoch == 0:
                print(f"Checkpoint save at epoch {epoch + 1}")
                training_network.save_network(os.path.join(output_folder, f"checkpoint_epoch_{epoch + 1}_save.pkl"))

            # early termination if no improvement has been seen in validation loss for a set number of epochs
            if epoch - early_stop_threshold > best_val_loss_epoch:
                print(f"\nTerminating training early - no improvement seen in {early_stop_threshold} epochs")
                break

            print()

        # save last network
        training_network.save_network(os.path.join(output_folder, "last.pkl"))

        # get total training time
        train_end_time = time.time() - train_start_time
        print(f"Training completed in {train_end_time / 3600} hours")
        
        print("Displaying stats graphs...")
            
        # get epoch numbers for plotting purposes
        epoch_nums = range(len(train_stats))

        # plot training loss per epoch
        train_loss_values = [epoch_stat.get_loss() for epoch_stat in train_stats]
        plt.plot(epoch_nums, train_loss_values)
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.title("Training Loss per Epoch")
        plt.show()

        # plot validation loss per epoch
        val_loss_values = [epoch_stat.get_loss() for epoch_stat in val_stats]
        plt.plot(epoch_nums, val_loss_values)
        plt.xlabel("Epochs")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss per Epoch")
        plt.show()

        # plot training time per epoch
        train_time_values = [epoch_stat.get_epoch_time() for epoch_stat in train_stats]
        plt.plot(epoch_nums, train_time_values)
        plt.xlabel("Epochs")
        plt.ylabel("Training Time (seconds)")
        plt.title("Training Time Elapsed per Epoch")
        plt.show()

        # plot validation time per epoch
        val_time_values = [epoch_stat.get_epoch_time() for epoch_stat in val_stats]
        plt.plot(epoch_nums, val_time_values)
        plt.xlabel("Epochs")
        plt.ylabel("Validation Time (seconds)")
        plt.title("Validation Time Elapsed per Epoch")
        plt.show()

        print("Writing stats to CSV...")

        # write train stats to a csv file
        with open(os.path.join(output_folder, 'train_stats.csv'), 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|')
            writer.writerow(['epoch_number', 'loss', 'epoch_time'])
            for i, train_stat in enumerate(train_stats):
                writer.writerow([i+1, train_stat.get_loss(), train_stat.get_epoch_time()])

        # write val stats to a csv file
        with open(os.path.join(output_folder, 'val_stats.csv'), 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|')
            writer.writerow(['epoch_number', 'loss', 'epoch_time'])
            for i, val_stat in enumerate(val_stats):
                writer.writerow([i+1, val_stat.get_loss(), val_stat.get_epoch_time()])

def _determine_epoch_learning_rate(epoch, num_epochs, initial_lr, final_lr):
    if num_epochs == 1:
        return initial_lr
    return initial_lr + epoch * ((final_lr - initial_lr) / (num_epochs - 1)) # num_epochs - 1 so that it cancels with epoch on the largest value of epoch

def _determine_kl_beta(epoch:int, initial_beta:float, max_kl_beta:float, increase_step:float):
    return float(min(initial_beta + epoch * increase_step, max_kl_beta))



if __name__ == '__main__':
    # Instantiate VAE
    vae = VariationalAutoencoder()

    # ENCODER
    # Layer 1
    layer_1 = RegularConvolutionalLayer(
        kernel_size=3,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        input_dims=(28, 28)
    )
    vae.append_layer(layer_1)
    # Layer 2
    layer_2 = RegularConvolutionalLayer(
        kernel_size=3,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        input_dims=(26, 26)
    )
    vae.append_layer(layer_2)
    # Layer 3
    layer_3 = RegularConvolutionalLayer(
        kernel_size=5,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        input_dims=(24, 24)
    )
    vae.append_layer(layer_3)
    # Layer 4
    layer_4 = RegularConvolutionalLayer(
        kernel_size=5,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        input_dims=(20, 20)
    )
    vae.append_layer(layer_4)
    # Layer 5
    layer_5 = RegularConvolutionalLayer(
        kernel_size=5,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        input_dims=(16, 16)
    )
    vae.append_layer(layer_5)
    # Layer 7
    layer_7 = FullyConnectedLayer(
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        num_inputs=12*12,
        num_outputs=8*8
    )
    vae.append_layer(layer_7)
    # Layer 8
    mean_layer = FullyConnectedLayer(
        activation_fn=VariationalAutoencoder.linear,
        activation_fn_dx=VariationalAutoencoder.linear_dx,
        num_inputs=8*8,
        num_outputs=4*4
    )
    log_var_layer = FullyConnectedLayer(
        activation_fn=VariationalAutoencoder.linear,
        activation_fn_dx=VariationalAutoencoder.linear_dx,
        num_inputs=8*8,
        num_outputs=4*4
    )
    layer_8 = SplitHeadFullyConnectedLayer(
        mean_layer=mean_layer,
        log_var_layer=log_var_layer
    )
    vae.append_layer(layer_8)

    # DECODER
    # Layer 9
    layer_9 = FullyConnectedLayer(
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        num_inputs=4*4,
        num_outputs=8*8,
    )
    vae.append_layer(layer_9)
    # Layer 10
    layer_10 = TransposedConvolutionalLayer(
        kernel_size=5,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        input_dims=(8, 8)
    )
    vae.append_layer(layer_10)
    # Layer 11
    layer_11 = TransposedConvolutionalLayer(
        kernel_size=5,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu,
        input_dims=(12, 12)
    )
    vae.append_layer(layer_11)
    # Layer 12
    layer_12 = TransposedConvolutionalLayer(
        kernel_size=5,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu,
        input_dims=(16, 16)
    )
    vae.append_layer(layer_12)
    # Layer 13
    layer_13 = TransposedConvolutionalLayer(
        kernel_size=5,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu,
        input_dims=(20, 20)
    )
    vae.append_layer(layer_13)
    # Layer 14
    layer_14 = TransposedConvolutionalLayer(
        kernel_size=3,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu,
        input_dims=(24, 24)
    )
    vae.append_layer(layer_14)
    # Layer 15
    layer_15 = TransposedConvolutionalLayer(
        kernel_size=3,
        activation_fn=VariationalAutoencoder.sigmoid,
        activation_fn_dx=VariationalAutoencoder.sigmoid_dx,
        input_dims=(26, 26)
    )
    vae.append_layer(layer_15)

    train_model(
        train_images_path='/home/troyxdp/Documents/University Work/Advanced Artificial Intelligence/Project/archive/train-images.idx3-ubyte',
        train_labels_path='/home/troyxdp/Documents/University Work/Advanced Artificial Intelligence/Project/archive/train-labels.idx1-ubyte',
        val_percentage=0.2,
        output_folder='/home/troyxdp/Documents/University Work/Advanced Artificial Intelligence/Project/networks/experiment_1',
        training_network=vae,
        num_epochs=20,
        initial_lr=0.00001,
        final_lr=0.000005,
        initial_kl_beta=0.1,
        final_kl_beta=1.0,
        kl_beta_step=0.1,
        beta_1=0.9,
        beta_2=0.999,
        clip_score=1.0,
        checkpoint_epoch=2,
        early_stop_threshold=10
    )