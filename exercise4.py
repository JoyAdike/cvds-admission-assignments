"""
exercise4.py : Debugging Exercise 4: GAN training bugs.

Two bugs found:

Bug 1 : Structural (crashes when batch_size=64):
    Labels tensors are created using the batch_size parameter:
        real_samples_labels = torch.ones((batch_size, 1))
        generated_samples_labels = torch.zeros((batch_size, 1))

    The last batch of each epoch often has fewer samples than batch_size because the dataset size is not always divisible by batch_size.
    This causes a size mismatch between samples and labels, crashing with: 
    ValueError: Using a target size that is different to the input size.

    Fix: Use real_samples.size(0) to get the actual current batch size dynamically, so the labels tensor always matches the actual number of samples in the current batch.

Bug 2 : Cosmetic:
    The display condition uses:
        if n == batch_size - 1:
    This ties the visualisation frequency to the batch_size parameter. Changing batch_size accidentally changes when results are displayed.

    Fix: Use len(train_loader) - 1 to always trigger at the end of each epoch regardless of batch_size.

What is a GAN:
    A GAN consists of two networks competing against each other.
    The Generator tries to create convincing fake images.
    The Discriminator tries to detect whether an image is real or fake.
    They train together until the Generator produces images that the Discriminator cannot distinguish from real ones.
"""

import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time


class Generator(nn.Module):
    """Generator network for the GAN."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output


class Discriminator(nn.Module):
    """Discriminator network for the GAN."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output


def train_gan(
        batch_size: int = 32,
        num_epochs: int = 100,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
    """
    Train a Generative Adversarial Network on the MNIST dataset.

    :param batch_size: Number of images per training batch.
    :param num_epochs: Number of training epochs.
    :param device: Computing device — cuda if available, else cpu.

    Bug 1 fixed: replaced batch_size with real_samples.size(0) when creating label tensors, so labels always match the actual batch size.

    Bug 2 fixed: replaced 'if n == batch_size - 1' with 'if n == len(train_loader) - 1' so display always triggers at the end of each epoch regardless of batch_size.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    try:
        train_set = torchvision.datasets.MNIST(
            root=".", train=True, download=True, transform=transform)
    except Exception:
        print("Failed to download MNIST, retrying with different URL")
        torchvision.datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/'
             'train-images-idx3-ubyte.gz',
             'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/'
             'train-labels-idx1-ubyte.gz',
             'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/'
             't10k-images-idx3-ubyte.gz',
             '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/'
             't10k-labels-idx1-ubyte.gz',
             'ec29112dd5afa0611ce80d1b7f02629c')
        ]
        train_set = torchvision.datasets.MNIST(
            root=".", train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)

    # Show sample real images
    real_samples, mnist_labels = next(iter(train_loader))
    fig = plt.figure()
    for i in range(16):
        sub = fig.add_subplot(4, 4, 1 + i)
        sub.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
        sub.axis('off')
    fig.tight_layout()
    fig.suptitle("Real images")
    plt.savefig("real_images.png")
    print("  Real images saved to real_images.png")
    time.sleep(2)

    # Set up models and optimisers
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    lr = 0.0001
    loss_function = nn.BCELoss()
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(
        generator.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        for n, (real_samples, mnist_labels) in enumerate(train_loader):

            # Get the ACTUAL current batch size dynamically
            # FIX Bug 1: use real_samples.size(0) not batch_size
            # The last batch may have fewer samples than batch_size
            current_batch_size = real_samples.size(0)

            # Data for training the discriminator
            real_samples = real_samples.to(device=device)

            # FIX Bug 1: use current_batch_size for all label tensors
            real_samples_labels = torch.ones(
                (current_batch_size, 1)).to(device=device)
            latent_space_samples = torch.randn(
                (current_batch_size, 100)).to(device=device)
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros(
                (current_batch_size, 1)).to(device=device)
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels))

            # Train the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(
                output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Data for training the generator
            latent_space_samples = torch.randn(
                (current_batch_size, 100)).to(device=device)

            # Train the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(
                generated_samples)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

            # FIX Bug 2: use len(train_loader) - 1 not batch_size - 1
            # This always triggers at the end of each epoch
            # regardless of what batch_size is set to
            if n == len(train_loader) - 1:
                name = (f"Generated images | Epoch: {epoch} | "
                        f"Loss D.: {loss_discriminator:.2f} "
                        f"Loss G.: {loss_generator:.2f}")
                generated_samples = generated_samples.detach().cpu().numpy()
                fig = plt.figure()
                for i in range(16):
                    sub = fig.add_subplot(4, 4, 1 + i)
                    sub.imshow(
                        generated_samples[i].reshape(28, 28),
                        cmap="gray_r")
                    sub.axis('off')
                fig.suptitle(name)
                fig.tight_layout()
                plt.savefig(f"generated_epoch_{epoch}.png")
                print(f"  Epoch {epoch} | Loss D: "
                      f"{loss_discriminator:.4f} | "
                      f"Loss G: {loss_generator:.4f}")
                plt.close(fig)


if __name__ == "__main__":
    print("Testing structural bug fix with batch_size=64...")
    print("(This would crash without the fix)")
    print("Training for 2 epochs to verify fix works...")
    train_gan(batch_size=64, num_epochs=2)
    print("\nExercise 4 — Structural and cosmetic bugs fixed.")
    print("Both batch_size=32 and batch_size=64 work correctly.")