# Facial Expression Recognition - Deep Learning
Project by Matteo Ingusci

## Motivations
The Facial Expression Recognition plays a crucial role in human-computer interactions.
The ability to interpret human emotions from facial expressions is essential to create an intelligent and empathetic system in various fields, like healthcare, education, and marketing.

Even if the field has been vastly explored, thanks also the raising of deep learning architectures, it still remains a challenge: minimal facial differences, imbalanced datasets, and moreover subjective labels.

This project aims to explore and compare a range of convolutional and transformer-based deep learning algorithms for the facial expression recognition task.
Each model will have a different configuration and will be trained to study the performance.

## Objectives

The main goal of this project is to evaluate and compare different deep learning models on the Facial Expression Recognition task over different metrics.
Thanks to the results, the project aims to identify the architectures that are most suitable for the task, taking into consideration some challenges, like the low resolution images, the class imbalance and the low number of samples.

## Project structure

- `setup_dataset.ipynb`: jupyter notebook to download and unzip the dataset.
- `train_ConvNet.py`: train the ConvNet model.
- `train_ResNet.py`: train the ResNet18 model.
- `train_VGG16.py`: train the VGG16 model.
- `train_GoogLeNet.py`: train the GoogLeNet model.
- `train_ViT-B16.py`: train the ViT-B/16 model.
- `train_ViT-mod.py`: train the ViT-B/16 with custom hyperparameters model.
- `test.ipynb`: test the models predicting three images.

Other files:

- `RAFDB_dataset.py`: contains the class that loads the dataset RAFDB,
- `train_function.py`: contains a function that trains the models. It is used by other files to train the models.
- `models/`: directory that contains many files. Each file has a function that allows to construct a model, or just a class.
- `results/`: directory that contains, and will contain, the results from the training runs.
