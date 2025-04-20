# CelebA WGAN-GP Implementation with PyTorch

This repository contains a PyTorch implementation of a Wasserstein GAN with Gradient Penalty (WGAN-GP) trained on the CelebA dataset to generate realistic face images.

## Description

The code defines a Generator and a Critic network based on Deep Convolutional GAN (DCGAN) architectures, adapted for the WGAN-GP framework. It includes a custom PyTorch `Dataset` for loading CelebA images, training loops, image generation, and model checkpointing.

## Key Features

*   WGAN-GP loss function for stable training.
*   DCGAN-style Generator and Critic architectures.
*   Custom `Dataset` for CelebA.
*   Gradient Penalty implementation for enforcing the Lipschitz constraint.
*   Training loop with configurable hyperparameters.
*   Image generation and saving during training.
*   Model checkpointing and resuming capabilities.

## Setup and Usage

1.  **Prerequisites:**
    *   Python 3.x
    *   PyTorch & Torchvision
    *   Pandas
    *   NumPy
    *   (Optional) Matplotlib/Tensorboard for visualization
    *   Ensure the CelebA dataset (`img_align_celeba` directory and `list_attr_celeba.csv`) is available at the specified path (e.g., `/kaggle/input/celeba-dataset/`).

2.  **Installation:**
    ```bash
    pip install torch torchvision pandas numpy
    ```

3.  **Configuration:**
    *   Adjust parameters like `IMAGE_SIZE`, `BATCH_SIZE`, `Z_DIM`, `LEARNING_RATE`, `EPOCHS`, etc., at the beginning of the script.
    *   Verify the `img_dir` and `attr_file` paths point to your CelebA dataset location.

4.  **Training:**
    *   Run the main Python script.
    *   Generated image samples will be saved in the `./output/` directory.
    *   Model checkpoints will be saved in the `./checkpoint/` directory.
    *   Final models will be saved in the `./models/` directory.

## Summary of Initial Code Changes & Fixes

The initial provided code required several critical modifications to enable effective training:

1.  **Enabled Full Training Duration:**
    *   **Change:** Removed the `break` statement at the end of the first epoch in the training loop.
    *   **Reason:** GANs, especially WGANs, require many epochs (tens to hundreds) to learn meaningful representations. Training for only one epoch resulted in random noise output as the model didn't have sufficient time to learn.

2.  **Corrected Data Normalization:**
    *   **Change:** Modified the `transforms.Normalize` step to use `mean=[0.5]*C` and `std=[0.5]*C`.
    *   **Reason:** The Generator uses `nn.Tanh()` as its final activation, outputting values in the `[-1, 1]` range. The input data normalization *must match* this range for the Critic to make valid comparisons between real and fake images. The previous ImageNet normalization (`mean=[0.485,...], std=[0.229,...]`) created a mismatch. Image saving logic was also updated (`* 0.5 + 0.5`) to correctly display images normalized to `[-1, 1]`.

3.  **Fixed Critic Output Layer:**
    *   **Change:** Removed the final `nn.LeakyReLU` activation *after* the last convolutional layer (`nn.Conv2d(512, 1, ...)` ) in the Critic network.
    *   **Reason:** The Critic in a WGAN/WGAN-GP should output a raw, unbounded score (criticism), not a value squashed by an activation function like LeakyReLU. This is essential for approximating the Wasserstein distance and for the Gradient Penalty calculation to work correctly.

4.  **Corrected Dataset Loading:**
    *   **Change:** Modified the `CBADataset.__getitem__` method to load images based on the `image_id` column from the `list_attr_celeba.csv` file, instead of assuming filenames are sequential numbers (`{:06d}.jpg`).
    *   **Reason:** The actual image filenames in `img_align_celeba` correspond to the `image_id` in the CSV, not simply the row index + 1. This ensures the correct images are loaded.

5.  **Enhanced Training Stability & Monitoring:**
    *   **Changes:** Increased default `EPOCHS`, slightly increased `CRITIC_STEPS`, added standard weight initialization (`weights_init`), improved logging of specific loss components (Wasserstein, GP), added saving of image grids using `make_grid`, refined checkpointing to include optimizer states and epoch number.
    *   **Reason:** These changes promote more stable training (initialization, critic steps), provide better insight into the training process (detailed logging, visual output), and allow for proper resumption of training (full checkpointing).

## Potential Future Work

While the above changes address critical issues, further improvements can be explored for better results or stability:

*   Extensive hyperparameter tuning (Learning Rate, Adam Betas, Batch Size, Critic Steps).
*   Applying Spectral Normalization, especially to the Critic.
*   Experimenting with different normalization layers (BatchNorm, LayerNorm, InstanceNorm).
*   Adjusting model capacity (number of filters).
*   Advanced monitoring using visualization libraries (TensorBoard/Matplotlib).
*   Adding data augmentation like random horizontal flips.
*   Utilizing mixed-precision training for speed and memory optimization.

## Dependencies

*   `torch`
*   `torchvision`
*   `pandas`
*   `numpy`
*   `os`
