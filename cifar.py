import argparse
import os
import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from deepspeed.accelerator import get_accelerator
from tqdm import tqdm

NUM_EPOCHS = 10
LEARNING_RATE = 1e-3


def add_argument():

    parser=argparse.ArgumentParser(description='CIFAR')

    # Data.
    # Cuda.
    parser.add_argument('--with_cuda', default=False, action='store_true',
                            help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                            help='whether use exponential moving average')

    # Train.
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                            help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

        # Include DeepSpeed configuration arguments.
    parser = deepspeed.add_config_arguments(parser)

    args=parser.parse_args()

    return args
def test(model_engine, test_dataloader, local_device):
    """Test the network on the test data.

    Args:
        model_engine (deepspeed.runtime.engine.DeepSpeedEngine): the DeepSpeed engine.
        test_dataloader (torch.utils.data.Dataset): the test dataset.
        local_device (str): the local device name.
    """
    # The 10 classes for CIFAR10.
    classes = (
        "plane", "car", "bird", "cat", "deer", "dog", "frog", 
        "horse", "ship", "truck"
    )

    # For total accuracy
    correct, total = 0, 0
    # For accuracy per class
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    # For calculating Precision, Recall, and F1 score
    true_positives = list(0.0 for i in range(10))
    false_positives = list(0.0 for i in range(10))
    false_negatives = list(0.0 for i in range(10))

    # Start testing.
    model_engine.eval()
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = model_engine(images.to(local_device))
            _, predicted = torch.max(outputs.data, 1)
            
            # Count the total accuracy
            total += labels.size(0)
            correct += (predicted == labels.to(local_device)).sum().item()

            # Count the accuracy and other metrics per class
            batch_correct = (predicted == labels.to(local_device)).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += batch_correct[i].item()
                class_total[label] += 1

                if predicted[i] == label:
                    true_positives[label] += 1
                else:
                    false_negatives[label] += 1
                    false_positives[predicted[i]] += 1

    # If master rank, print results
    if model_engine.local_rank == 0:
        accuracy = 100 * correct / total
        print(f"Accuracy of the network on the {total} test images: {accuracy: .0f} %")

        # For each class, print the accuracy
        for i in range(10):
            if class_total[i] > 0:
                class_accuracy = 100 * class_correct[i] / class_total[i]
                print(f"Accuracy of {classes[i]: >5s} : {class_accuracy:2.0f} %")

                # Calculate Precision, Recall, and F1 Score per class
                precision = true_positives[i] / (true_positives[i] + false_positives[i] + 1e-8)
                recall = true_positives[i] / (true_positives[i] + false_negatives[i] + 1e-8)
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

                print(f"Precision of {classes[i]: >5s} : {precision * 100:2.0f} %")
                print(f"Recall of {classes[i]: >5s} : {recall * 100:2.0f} %")
                print(f"F1 Score of {classes[i]: >5s} : {f1_score * 100:2.0f} %")

        # Calculate and print the average F1 score
        avg_f1_score = sum(f1_score for f1_score in true_positives) / 10
        print(f"Average F1 Score: {avg_f1_score * 100:2.0f} %")



def get_ds_config(args):
    """Get the DeepSpeed configuration dictionary."""
    ds_config = {
    "train_batch_size": 4,
    "optimizer": {
        "type": "Adam",
        "params": {
        "lr": 0.001,
        "betas": [
            0.8,
            0.999
        ],
        "eps": 1e-8,
        "weight_decay": 3e-7
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
        "warmup_min_lr": 0,
        "warmup_max_lr": 0.001,
        "warmup_num_steps": 1000
        }
    },
    "logging": {
            "level": "WARN"
        }
    }

    return ds_config

def train_epoch(epoch: int, model: nn.Module, criterion: nn.Module, train_dataloader: DataLoader, local_device, model_engine):
    model.train()
    with tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}]", disable=not deepspeed.comm.get_rank() == 0) as pbar:
        for images, labels in pbar:
            inputs, labels = images.to(local_device), labels.to(local_device)
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            pbar.set_postfix({"loss": loss.item()})

def main(args):
    # Initialize DeepSpeed distributed backend.
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK"))
    get_accelerator().set_device(local_rank)

    ########################################################################
    # Step 1: Data Preparation
    ########################################################################
    transform_train = transforms.Compose(
        [transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()]
    )
    transform_test = transforms.ToTensor()

    if torch.distributed.get_rank() != 0:
        # Rank 0 will download data first
        torch.distributed.barrier()

    # Load CIFAR data
    train_dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True,
            download=True,
            transform=transform_train
        )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',  
        train=False,
        download=True,
        transform=transform_test
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=200,
        shuffle=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False
    )

    if torch.distributed.get_rank() == 0:
        # Indicate other ranks can proceed
        torch.distributed.barrier()

    ########################################################################
    # Step 2: Define the network with DeepSpeed
    ########################################################################
    net = torchvision.models.resnet18(num_classes=10)
   
    # Initialize DeepSpeed to use the following features:
    #   1) Distributed model.
    #   2) Distributed data loader.
    #   3) DeepSpeed optimizer.
    ds_config = get_ds_config(args)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=net.parameters(),
        training_data=train_dataset,  # Pass the Dataset here
        config=ds_config,
    )

    # Get the local device name (str) and local rank (int)
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    # Define the Classification Cross-Entropy loss function
    criterion = nn.CrossEntropyLoss()

    ########################################################################
    # Step 3: Train the network
    ########################################################################
    start_epoch = 0
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_epoch(epoch, net, criterion, train_dataloader, local_device, model_engine)
    print("Finished Training")

    ########################################################################
    # Step 4: Test the network on the test data
    ########################################################################
    # Implement testing functionality here if needed.
    test(model_engine, test_dataloader, local_device)
if __name__ == "__main__":
    args = add_argument()
    main(args)
