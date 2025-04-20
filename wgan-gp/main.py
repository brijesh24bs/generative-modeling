# necessary libraries
import os
import numpy as np
import pandas as pd
import time # Added for timing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
from torchvision.utils import save_image, make_grid # Added make_grid for better visualization
from IPython.display import Image, display

# Display initial image (no changes needed here)
try:
    display(Image("/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/000001.jpg"))
except FileNotFoundError:
    print("Warning: Initial display image not found. Skipping display.")

# parameters
IMAGE_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 128 # Reduced batch size slightly, maybe helpful for memory/stability
NUM_FEATURES = 64 # This parameter doesn't seem used in the model definitions
Z_DIM = 128
LEARNING_RATE = 0.0002
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.999
EPOCHS = 50 # Increased epochs significantly - adjust based on your time/compute
CRITIC_STEPS = 5 # Slightly increased critic steps, common practice
GP_WEIGHT = 10.0
LOAD_MODEL = False
SAVE_FREQ = 5 # Save more frequently initially
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Dataset Class (No changes needed) ---
class CBADataset(Dataset):
    def __init__(self, img_dir, attr_file, transform=None):
        super().__init__()
        self.img_dir = img_dir
        # Check if attribute file exists
        if not os.path.isfile(attr_file):
             raise FileNotFoundError(f"Attribute file not found: {attr_file}")
        try:
            self.attr = pd.read_csv(attr_file)
        except Exception as e:
            print(f"Error reading attribute file {attr_file}: {e}")
            # Attempt to list directory contents for debugging
            print(f"Contents of /kaggle/input/celeba-dataset/: {os.listdir('/kaggle/input/celeba-dataset/')}")
            raise e

        # Check if image directory exists
        if not os.path.isdir(img_dir):
            raise NotADirectoryError(f"Image directory not found: {img_dir}")
        self.img_paths = [os.path.join(img_dir, fname) for fname in self.attr['image_id']]
        # Verify a few image paths
        if len(self.img_paths) > 0:
             if not os.path.isfile(self.img_paths[0]):
                 print(f"Warning: First image file not found: {self.img_paths[0]}")
                 # Try listing the image directory contents
                 try:
                     print(f"Contents of {img_dir}: {os.listdir(img_dir)[:10]}...") # Show first 10
                 except FileNotFoundError:
                     print(f"Could not list contents of {img_dir}")

        self.transform = transform

    def __len__(self):
        return len(self.attr)

    def __getitem__(self, idx):
        # Original code used idx+1 for filename, but CSV uses actual filenames.
        # Let's use the image_id column from the CSV.
        img_filename = self.attr.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_filename)
        try:
            image = read_image(img_path)
        except Exception as e:
             print(f"Error reading image {img_path} at index {idx}: {e}")
             # Return a dummy tensor or handle appropriately
             # For now, re-raise to stop execution if an image is missing/corrupt
             raise e

        if self.transform:
            # Apply transforms. Ensure ToPILImage is used if needed by subsequent transforms
            # but read_image already returns a Tensor. If Resize needs PIL, insert ToPILImage.
            # Current transform expects Tensor input for Normalize.
             image = self.transform(image)
        return image


# --- Transforms ---
# **ISSUE 2 FIX:** Changed Normalization to [-1, 1] range
transform = transforms.Compose([
    transforms.ToPILImage(), # read_image gives Tensor, Resize needs PIL or Tensor depending on version/backend
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True), # Use antialias=True for better quality downsampling
    transforms.CenterCrop(IMAGE_SIZE), # Ensure exact size after resize potentially
    transforms.ToTensor(), # Converts PIL Image (0-255) or Tensor to FloatTensor (0.0-1.0)
    transforms.Normalize([0.5] * CHANNELS, [0.5] * CHANNELS), # Scales (0.0-1.0) to (-1.0-1.0)
])

# --- Data Loading ---
img_dir = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba"
attr_file = "/kaggle/input/celeba-dataset/list_attr_celeba.csv"

try:
    dataset = CBADataset(img_dir, attr_file, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True) # Added num_workers, pin_memory
    print(f"Dataset length: {len(dataset)}")

    # Test dataloader
    print("Testing dataloader...")
    sample_batch = next(iter(dataloader))
    print(f"Sample batch size: {sample_batch.size()}")
    print(f"Sample batch min/max: {sample_batch.min():.2f}/{sample_batch.max():.2f}") # Should be close to -1/1
    display(make_grid(sample_batch[:16] * 0.5 + 0.5, nrow=4)) # Display a grid of real samples

except (FileNotFoundError, NotADirectoryError, Exception) as e:
    print(f"Error initializing dataset or dataloader: {e}")
    # Optionally exit or handle error
    raise SystemExit("Could not load data, stopping execution.")


# --- Critic Model ---
class Critic(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Simplified architecture slightly, removed dropout for now
        self.conv_layers = nn.Sequential(
            # Input: BATCH_SIZE x channels x 64 x 64
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False), # -> 64x32x32
            nn.LeakyReLU(0.2, inplace=True),
            # Removed Dropout

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # -> 128x16x16
            nn.InstanceNorm2d(128, affine=True), # Using InstanceNorm instead of BatchNorm/Dropout
            nn.LeakyReLU(0.2, inplace=True),
            # Removed Dropout

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), # -> 256x8x8
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Removed Dropout

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False), # -> 512x4x4
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Removed Dropout

            # **ISSUE 3 FIX:** Removed LeakyReLU from the final layer
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0), # -> 1x1x1
            # No LeakyReLU here
            nn.Flatten() # -> scalar output per image
         )

    def forward(self, x):
        x = self.conv_layers(x)
        return x

# Test Critic
print("Testing Critic...")
critic = Critic(CHANNELS)
critic_output = critic(torch.randn(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
print(f"Critic output size: {critic_output.size()}") # Should be [BATCH_SIZE, 1]


# --- Generator Model ---
class Generator(nn.Module):
    def __init__(self, z_dim, channels):
        super().__init__()
        self.z_dim = z_dim
        self.channels = channels

        self.main = nn.Sequential(
            # Input: BATCH_SIZE x z_dim x 1 x 1
            nn.ConvTranspose2d(self.z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False), # -> 512x4x4
            nn.BatchNorm2d(512), # Removed momentum=0.9, use default
            nn.ReLU(True), # Using ReLU instead of LeakyReLU here, common practice

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False), # -> 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False), # -> 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False), # -> 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, self.channels, kernel_size=4, stride=2, padding=1, bias=False), # -> channelsx64x64
            nn.Tanh() # Output [-1, 1] - matches new normalization
        )

    def forward(self, x):
        x = x.view(-1, self.z_dim, 1, 1)
        x = self.main(x)
        return x

# Test Generator
print("Testing Generator...")
generator = Generator(Z_DIM, CHANNELS)
generator_output = generator(torch.randn(BATCH_SIZE, Z_DIM))
print(f"Generator output size: {generator_output.size()}") # Should be [BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE]


# --- WGANGP Class (Compile method modified slightly for clarity) ---
class WGANGP(nn.Module):
    def __init__(self, critic, generator, latent_dim, critic_steps, gp_weight):
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

    def compile(self, g_optimizer, c_optimizer):
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        # Moving optimizer state to device is handled implicitly if models are already on device
        # Or can be done explicitly if needed, but let's keep it simple for now.

    def gradient_penalty(self, real_images, fake_images, device):
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
        pred = self.critic(interpolated)

        gradients = torch.autograd.grad(
            outputs=pred,
            inputs=interpolated,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True # More efficient
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self, real_images, device):
        batch_size = real_images.size(0)

        # --- Train Critic ---
        for _ in range(self.critic_steps):
            self.c_optimizer.zero_grad()

            random_latent_vectors = torch.randn(batch_size, self.latent_dim, device=device)
            with torch.no_grad(): # Don't need gradients for generator when training critic
                 fake_images = self.generator(random_latent_vectors).detach()

            fake_predictions = self.critic(fake_images)
            real_predictions = self.critic(real_images)

            c_wass_loss = torch.mean(fake_predictions) - torch.mean(real_predictions)
            c_gp = self.gradient_penalty(real_images, fake_images, device)
            c_loss = c_wass_loss + c_gp * self.gp_weight

            c_loss.backward()
            self.c_optimizer.step()

        # --- Train Generator ---
        self.g_optimizer.zero_grad()

        # Generate new fake images for generator update
        random_latent_vectors = torch.randn(batch_size, self.latent_dim, device=device)
        fake_images = self.generator(random_latent_vectors)
        fake_predictions = self.critic(fake_images) # Critic processes these new fakes

        # We want the critic to think the fake images are real -> maximize fake_predictions
        # Maximizing fake_predictions is equivalent to minimizing -fake_predictions
        g_loss = -torch.mean(fake_predictions)

        g_loss.backward()
        self.g_optimizer.step()

        # Return detached scalar values for logging
        return c_loss.item(), c_wass_loss.item(), c_gp.item(), g_loss.item()


# --- Initialization and Training ---
critic = Critic(CHANNELS).to(DEVICE)
generator = Generator(Z_DIM, CHANNELS).to(DEVICE)

# Apply weight initialization (often helps convergence)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

print("Initializing weights...")
critic.apply(weights_init)
generator.apply(weights_init)

wgangp = WGANGP(
    critic=critic,
    generator=generator,
    latent_dim=Z_DIM,
    critic_steps=CRITIC_STEPS,
    gp_weight=GP_WEIGHT,
)
wgangp.to(DEVICE) # Ensure the WGANGP module itself doesn't hold parameters needing device transfer

# Check if LOAD_MODEL is True and checkpoint exists
start_epoch = 1
if LOAD_MODEL:
    checkpoint_path = "./checkpoint/latest_checkpoint.pth" # Example path
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        # Optimizers need to be created BEFORE loading their state dicts
        c_optimizer = optim.Adam(wgangp.critic.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA_1, ADAM_BETA_2))
        g_optimizer = optim.Adam(wgangp.generator.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA_1, ADAM_BETA_2))
        c_optimizer.load_state_dict(checkpoint['c_optimizer_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("LOAD_MODEL is True, but checkpoint not found. Starting from scratch.")
        LOAD_MODEL = False # Reset flag if file not found

# Initialize optimizers if not loaded from checkpoint
if not LOAD_MODEL:
    c_optimizer = optim.Adam(wgangp.critic.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA_1, ADAM_BETA_2))
    g_optimizer = optim.Adam(wgangp.generator.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA_1, ADAM_BETA_2))

wgangp.compile(g_optimizer=g_optimizer, c_optimizer=c_optimizer)

# Create directories
os.makedirs("./checkpoint", exist_ok=True)
os.makedirs("./output", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# Fixed noise for consistent visualization across epochs
fixed_noise = torch.randn(64, Z_DIM, device=DEVICE) # Generate 64 images for visualization grid

print("Starting Training Loop...")
total_start_time = time.time()

# --- Training Loop ---
for epoch in range(start_epoch, EPOCHS + 1):
    epoch_start_time = time.time()
    critic_losses, gen_losses = [], []
    critic_wass_losses, critic_gps = [], [] # For more detailed logging

    # Set models to train mode
    wgangp.train()

    for batch_idx, real_images in enumerate(dataloader):
        real_images = real_images.to(DEVICE)
        c_loss, c_wass, c_gp, g_loss = wgangp.train_step(real_images, DEVICE)

        critic_losses.append(c_loss)
        gen_losses.append(g_loss)
        critic_wass_losses.append(c_wass)
        critic_gps.append(c_gp)

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx+1}/{len(dataloader)}] \t"
                  f"C Loss: {c_loss:.4f} (Wass: {c_wass:.4f}, GP: {c_gp:.4f}) \t"
                  f"G Loss: {g_loss:.4f}")

    # Calculate average losses for the epoch
    avg_c_loss = np.mean(critic_losses)
    avg_g_loss = np.mean(gen_losses)
    avg_c_wass = np.mean(critic_wass_losses)
    avg_c_gp = np.mean(critic_gps)
    epoch_time = time.time() - epoch_start_time

    print("-" * 50)
    print(f"Epoch {epoch}/{EPOCHS} Summary:")
    print(f"Avg Critic Loss: {avg_c_loss:.4f} (Avg Wass: {avg_c_wass:.4f}, Avg GP: {avg_c_gp:.4f})")
    print(f"Avg Generator Loss: {avg_g_loss:.4f}")
    print(f"Time: {epoch_time:.2f}s")
    print("-" * 50)

    # --- Generate and Save Images ---
    wgangp.eval() # Set models to evaluation mode for generation
    with torch.no_grad():
        generated_images = generator(fixed_noise).cpu() # Generate on device, move to CPU for saving
        # Save a grid of generated images
        # The normalization is [-1, 1], so scale back to [0, 1] for saving/display
        img_grid = make_grid(generated_images * 0.5 + 0.5, nrow=8)
        save_image(img_grid, f'./output/generated_epoch_{epoch:03d}.png')
        # Display the grid in environments like Kaggle notebooks
        if epoch % SAVE_FREQ == 0 or epoch == EPOCHS:
             display(Image(filename=f'./output/generated_epoch_{epoch:03d}.png'))


    # --- Save Checkpoints ---
    if epoch % SAVE_FREQ == 0 or epoch == EPOCHS:
        checkpoint_path = f"./checkpoint/checkpoint_epoch_{epoch}.pth"
        latest_checkpoint_path = "./checkpoint/latest_checkpoint.pth"
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'c_optimizer_state_dict': c_optimizer.state_dict(),
            'fixed_noise': fixed_noise # Optional: save fixed noise
        }, checkpoint_path)
        # Overwrite/create a 'latest' checkpoint file for easy resuming
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'c_optimizer_state_dict': c_optimizer.state_dict(),
            'fixed_noise': fixed_noise
        }, latest_checkpoint_path)

        print(f"Checkpoint saved to {checkpoint_path} and {latest_checkpoint_path}")

    # **ISSUE 1 FIX:** Removed the break statement
    # break

# --- Save Final Models ---
torch.save(generator.state_dict(), "./models/generator_final.pth")
torch.save(critic.state_dict(), "./models/critic_final.pth")
print("Final models saved.")
total_time = time.time() - total_start_time
print(f"Total Training Time: {total_time/60:.2f} minutes")
