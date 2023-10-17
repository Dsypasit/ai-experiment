import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

def create_image_data_loaders(data_folder, batch_size=32, preprocess=None):
    # Define preprocessing transforms
    if preprocess == None:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Load dataset
    dataset = datasets.ImageFolder(root=data_folder, transform=preprocess)

    # Get the length of the dataset
    dataset_size = len(dataset)

    # Define the proportions for training and test sets
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for training and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, train_dataset.dataset.classes
