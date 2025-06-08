# coding=utf-8
from torchvision import transforms
import numpy as np
import os

def act_train():
    return transforms.Compose([
        transforms.ToTensor()
    ])


def act_test():
    return transforms.Compose([
        transforms.ToTensor()
    ])


def loaddata_from_numpy(dataset='dsads', root_dir='./data/act/'):
    x = np.load(root_dir+dataset+'/'+dataset+'_x.npy')
    ty = np.load(root_dir+dataset+'/'+dataset+'_y.npy')
    cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]
    return x, cy, py, sy


def accumulate_participant_files(args, data_dir, users_list):
    data = []
    labels = []

    for participant in users_list:
        print(f"Processing participant {participant}")
        file_name = f"P{participant:03}.data"
        file_path = os.path.join(data_dir, args.dataset, file_name)
        if not os.path.isfile(file_path):
            print("skipping")
            continue  

        try:
            participant_data = np.load(file_path, allow_pickle=True)
            windows, activity_values, user_values = participant_data 
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue

        for window, activity, user in zip(windows, activity_values, user_values):
            window_data = window.to_numpy(dtype=np.float32)
            usable_length = (window_data.shape[0] // args.input_shape[2]) * args.input_shape[2]
            if usable_length == 0:
                continue  
            
            window_data = window_data[:usable_length, :]
            reshaped_data = window_data.reshape(-1, args.input_shape[2], window_data.shape[1])  # (windows, seq_len, features)

            reshaped_data = reshaped_data.transpose(0, 2, 1)  

            activity_label = np.full((reshaped_data.shape[0],), activity, dtype=np.int32)
            user_label = np.full((reshaped_data.shape[0],), participant, dtype=np.int32)
            sensor_label = np.zeros((reshaped_data.shape[0],), dtype=np.int32)  

            data.append(reshaped_data)
            labels.append(np.stack((activity_label, user_label, sensor_label), axis=1))  

    if data:
        x = np.concatenate(data, axis=0).astype(np.float32)  
        ty = np.concatenate(labels, axis=0).astype(np.int32)  
        cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]

        print(f"Data shape: {x.shape}, Labels shape: {ty.shape}")
        return x, cy, py, sy
    else:
        print("No valid data processed.")
        return None, None, None, None
