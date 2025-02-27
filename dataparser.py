from scipy.io import wavfile
import wave, os, glob
import numpy as np
import matplotlib.pyplot as plt
import torch


def audio_parser(filepath_clean, filepath_dist, buffer_size):
    # This function reads audio files from folders, discards one of the channels,
    # and batches them into batches of 512.
    clean_buffers = []
    dist_buffers = []
    print("we're in..")
    for filename in glob.glob(os.path.join(filepath_clean, '*.wav')):
        print("FILENAME: ", filename)
        name = filename.split("/")[7].split("_")
        print("NAME: ", name)
        # Use the name to get the parameter values from the mixing table
        trim, high, mid, skew, low = int(name[1][1:]), int(name[2][1:]), int(name[3][1:]), int(name[4][4:]), int(name[5][1:].split(".")[0]), 
        id = name[0][1]
        settings = "_".join(name[1:])
        _, data_clean = wavfile.read(filename)
        data_clean = data_clean[:, 0]        # Take only first column, they are identical
        _, data_dist = wavfile.read(filepath_dist + "/D" + id + "_" + settings)
        data_dist = data_dist[:, 0]         # We take first row, idk why let's try it

        print("shape data clean: ", np.shape(data_clean))
        print("shape data dist: ", np.shape(data_dist))

        # Fix the batches of clean data
        clean_chunks = [data_clean[i:i + buffer_size] for i in range(0, len(data_clean), buffer_size)]
        # We want to make rows of size 512 for each parameter, so we end up with a matrix of size 6x512, first row is audio, rest are parameters
        row = np.ones(512)
        parameters = [row*trim, row*high, row*mid, row*skew, row*low]
        clean_parameter_chunks = []
        for chunk in clean_chunks[:-1]:
            chunk = np.vstack([chunk, parameters])
            clean_parameter_chunks.append(chunk.T)
        clean_chunks = np.array(clean_parameter_chunks)

        dist_chunks = [data_dist[i:i + buffer_size] for i in range(0, len(data_dist), buffer_size)]

        print("shape clean buffer chunks: ", np.shape(clean_chunks))
        print("shape dist buffer chunks: ", np.shape(dist_chunks[:-1]))
      
    # Return everything except the last chunk of dist_chunks which may not be of size 512
    return clean_chunks, np.array(dist_chunks[:-1])


#data, targets = audio_parser("Training Data/C", "Training Data/D", 512)

