import os
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from RAFDB_dataset import RAFDBDataset

from torchvision import transforms as T

# import tensorboard writer
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, top_k_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train(model, model_name, image_size=(224, 224), lr=0.001, scheduler=False, epochs=100, patience=10):

    # if not exists, create results directory
    if not os.path.exists(f'./results/{model_name}'):
        os.makedirs(f'./results/{model_name}')

    # Define transformations for training and validation datasets
    transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2), 
        T.Resize(image_size),
    ])
    transforms_val = T.Compose([
        T.Resize(image_size)
    ])

    # define the train and validation datasets
    train_dataset = RAFDBDataset('./raf-db-dataset', custom_transform=transforms)
    val_dataset = RAFDBDataset('./raf-db-dataset', custom_transform=transforms_val, train=False)

    # defne a weighted random sampler to handle class imbalance
    train_sampler = WeightedRandomSampler(
        [train_dataset.class_weights[label] for (_, label) in train_dataset],
        num_samples=len(train_dataset),
        replacement=True
    )

    # Create DataLoaders for training and validation datasets
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
    )

    # find the device to use and move the model to that device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function, optimizer, and learning rate scheduler (if desired)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if scheduler:
        scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Initialize TensorBoard writer and variables for tracking best model
    writer = SummaryWriter(log_dir=f'./results/{model_name}')
    best_f1_score = 0.0
    best_epoch = 0
    patience_counter = 0

    # For each epoch
    for epoch in trange(epochs, desc="Training Epochs"):

        # traning phase
        model.train()
        # Iterate over training data
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            # move images and labels to the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            # Compute loss
            loss = loss_function(outputs, labels)

            # Backward and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        # init variables
        total_loss = 0.0
        all_preds = []
        all_preds_scores = []
        all_labels = []
        # disable gradient calculation for validation
        with torch.no_grad():
            # Iterate over validation data
            for images, labels in tqdm(val_dataloader, desc="Validation", leave=False):
                # move images and labels to the device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                # Compute loss
                loss = loss_function(outputs, labels)
                # accumulate loss
                total_loss += loss.item()

                # Get predictions
                _, predicted = torch.max(outputs, 1)
                # store predictions and labels
                all_preds_scores.extend(outputs.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # average the total loss
        total_loss /= len(val_dataloader)

        # If using a scheduler, step it with the total loss
        if scheduler:
            scheduler_fn.step(total_loss)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Calculate top-k accuracy
        top_1_accuracy = top_k_accuracy_score(all_labels, all_preds_scores, k=1)
        top_2_accuracy = top_k_accuracy_score(all_labels, all_preds_scores, k=2)
        top_3_accuracy = top_k_accuracy_score(all_labels, all_preds_scores, k=3)

        # compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=train_dataset.class_names, yticklabels=train_dataset.class_names, cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.tight_layout()

        # log metrics to TensorBoard
        writer.add_scalar('Loss/val', total_loss, epoch)
        writer.add_scalar('Accuracy/val', accuracy, epoch)
        writer.add_scalar('Precision/val', precision, epoch)
        writer.add_scalar('Recall/val', recall, epoch)
        writer.add_scalar('F1/val', f1, epoch)

        writer.add_scalar('Top-1 Accuracy/val', top_1_accuracy, epoch)
        writer.add_scalar('Top-2 Accuracy/val', top_2_accuracy, epoch)
        writer.add_scalar('Top-3 Accuracy/val', top_3_accuracy, epoch)

        writer.add_figure('ConfusionMatrix/val', fig, global_step=epoch)

        # If a new best F1 score is achieved, save the model and update the best epoch
        if f1 > best_f1_score:
            best_f1_score = f1
            patience_counter = 0
            torch.save(model.state_dict(), f'./results/{model_name}/best_model.pth')
            best_epoch = epoch
        else:
            # Increment patience counter if no improvement
            patience_counter += 1

        writer.add_scalar('Best Epoch/val', best_epoch + 1, epoch)

        # Check for early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best F1 score: {best_f1_score:.4f} at epoch {best_epoch + 1}.")
            break

    print(f"Training complete. Best F1 score: {best_f1_score:.4f} at epoch {best_epoch + 1}.")
    writer.close()