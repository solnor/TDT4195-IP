import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import utils
import dataloaders
import torchvision
from trainer import Trainer
torch.random.manual_seed(0)
np.random.seed(0)


# Load the dataset and print some stats
batch_size = 64

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])])

dataloader_train, dataloader_test = dataloaders.load_dataset(
    batch_size, image_transform)
example_images, _ = next(iter(dataloader_train))
print(f"The tensor containing the images has shape: {example_images.shape} (batch size, number of color channels, height, width)",
      f"The maximum value in the image is {example_images.max()}, minimum: {example_images.min()}", sep="\n\t")


def create_model():
    """
        Initializes the mode. Edit the code below if you would like to change the model.
    """
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        #
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        #
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        #
        nn.Flatten(),
        nn.Linear(128*4*4, 64), # Output shape of last MaxPool2d is 4x4 with 128 channels
        nn.ReLU(),
        nn.Linear(64, 10),
        # No need to include softmax, as this is already combined in the loss function
    )
    # Transfer model to GPU memory if a GPU is available
    model = utils.to_cuda(model)
    return model


model_SGD = create_model()


# Test if the model is able to do a single forward pass
example_images = utils.to_cuda(example_images)
output = model_SGD(example_images)
print("Output shape:", output.shape)
expected_shape = (batch_size, 10)  # 10 since mnist has 10 different classes
assert output.shape == expected_shape,    f"Expected shape: {expected_shape}, but got: {output.shape}"


# Hyperparameters
learning_rate = .02
learning_rate = .001
num_epochs = 5


# Use CrossEntropyLoss for multi-class classification
loss_function = torch.nn.CrossEntropyLoss()

# Define optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(model_SGD.parameters(),
                            lr=learning_rate)


trainer = Trainer(
    model=model_SGD,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer
)
train_loss_dict_SGD, test_loss_dict_SGD = trainer.train(num_epochs)

model_Adam = create_model()

# Define optimizer (Adam)
learning_rate = .001
optimizer = torch.optim.Adam(model_Adam.parameters(),
                             lr=learning_rate)

trainer = Trainer(
    model=model_Adam,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer
)
train_loss_dict_Adam, test_loss_dict_Adam = trainer.train(num_epochs)

# We can now plot the training loss with our utility script

# Plot loss
utils.plot_loss(train_loss_dict_SGD, label="Train Loss SGD")
utils.plot_loss(test_loss_dict_SGD, label="Test Loss SGD")
utils.plot_loss(train_loss_dict_Adam, label="Train Loss Adam")
utils.plot_loss(test_loss_dict_Adam, label="Test Loss Adam")
# Limit the y-axis of the plot (The range should not be increased!)
plt.ylim([0, .1])
plt.legend()
plt.xlabel("Global Training Step")
plt.ylabel("Cross Entropy Loss")
plt.savefig(utils.image_output_dir.joinpath("task2a_plot.png"))

final_loss_SGD, final_acc_SGD = utils.compute_loss_and_accuracy(
    dataloader_test, model_SGD, loss_function)
print(f"Final Test loss SGD: {final_loss_SGD}. Final Test accuracy SGD: {final_acc_SGD}")

final_loss_Adam, final_acc_Adam = utils.compute_loss_and_accuracy(
    dataloader_test, model_Adam, loss_function)
print(f"Final Test loss Adam: {final_loss_Adam}. Final Test accuracy Adam: {final_acc_Adam}")
plt.show()
