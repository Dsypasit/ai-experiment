import torch
from torch import nn, optim

# Assuming train_dataloader and test_dataloader are defined
# Assuming refit_model and criterion are defined
# Assuming optimizer is defined

import torch

def train(refit_model, optimizer, criterion, train_dataloader, test_dataloader, epochs=10, device=None):
    # Set the device to GPU if available
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    refit_model.to(device)

    losses = []
    accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        train_total = 0
        running_loss = 0.0
        correct = 0

        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = refit_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            batch_size = labels.size(0)
            correct += (predicted == labels).sum().item()
            train_total += batch_size

            # Update running loss
            running_loss += loss.item()

            if i % 10 == 0:  # Print statistics every 10 mini-batches
                print(f'Epoch [{epoch + 1}], Step [{i + 1}], Loss: {running_loss / (i + 1):.3f}')

        # Store epoch metrics for training
        epoch_train_accuracy = 100 * correct / train_total
        epoch_train_loss = running_loss / len(train_dataloader)
        losses.append(epoch_train_loss)
        accuracies.append(epoch_train_accuracy)

        # Testing phase
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():  # In evaluation mode, no need to compute gradients
            for i, data in enumerate(test_dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = refit_model(inputs)
                loss = criterion(outputs, labels)

                # Compute test loss
                test_loss += loss.item()

                # Compute test accuracy
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_accuracy = 100 * test_correct / test_total
        test_loss = test_loss / len(test_dataloader)  # Average test loss

        # Store epoch metrics for testing
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Epoch [{epoch + 1}], Train Loss: {epoch_train_loss:.3f}, Train Accuracy: {epoch_train_accuracy:.2f}%, Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%')

    return losses, accuracies, test_losses, test_accuracies
