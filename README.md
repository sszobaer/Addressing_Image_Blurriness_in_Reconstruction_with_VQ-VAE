# Addressing_Image_Blurriness_in_Reconstruction_with_VQ-VAE

This repository demonstrates the use of **Vector Quantized Variational Autoencoders (VQ-VAE)** for image reconstruction. The model is built and trained using PyTorch to reconstruct images based on a dataset of your choice. The goal is to perform unsupervised learning for image representation and generation.

## Table of Contents

1. [Introduction](#introduction)
2. [Model Overview](#model-overview)
3. [Setup and Installation](#setup-and-installation)
4. [Customizing the Model](#customizing-the-model)
5. [Training the Model](#training-the-model)
6. [Simulating the Model](#simulating-the-model)
7. [Visualizing Results](#visualizing-results)
8. [Additional Considerations](#additional-considerations)
9. [Contributing](#contributing)
10. [License](#license)

---

## Introduction

This project focuses on reconstructing images from a dataset using the **VQ-VAE (Vector Quantized Variational Autoencoder)** architecture. The model is trained on a set of images and is evaluated based on how accurately it reconstructs the original images. It uses a discrete latent space and quantization for better image representation and generation.

## Model Overview

**VQ-VAE (Vector Quantized Variational Autoencoder)** is an autoencoder-based architecture where the encoder maps input images to discrete latent codes. These latent codes are learned from a fixed codebook, which is used by the decoder to reconstruct the images. The key advantage of VQ-VAE is that it combines the benefits of VAEs with the use of vector quantization to improve the quality of the learned representations.

### Key Components:
- **Encoder**: Maps input images to a latent space of fixed-length vectors.
- **Quantization**: Discretizes the latent vectors by mapping them to a finite set of embeddings (codebook vectors).
- **Decoder**: Reconstructs images from the quantized latent vectors.

## Setup and Installation

### Prerequisites:
- Python 3.6+
- PyTorch
- Matplotlib (for visualization)
- NumPy
- CUDA (for GPU acceleration, optional)

### Installing Required Libraries:

You can install the required libraries by running:

```bash
pip install torch torchvision matplotlib numpy
```

### Cloning the Repository:

To clone this repository:

```bash
git clone https://github.com/sszobaer/Addressing_Image_Blurriness_in_Reconstruction_with_VQ-VAE.git
cd vq-vae-image-reconstruction
```

---

## Customizing the Model

You can customize the model's architecture by modifying the parameters passed when initializing the VQ-VAE model. Below are some key parameters you can customize:

- **`input_channels`**: Number of input channels in the images (e.g., 3 for RGB images).
- **`num_embeddings`**: Number of entries in the codebook (the size of the latent space).
- **`embedding_dim`**: The dimensionality of each latent vector (the size of the codebook entries).
- **`hidden_dim`**: The number of hidden units in the encoder and decoder.
- **`num_epochs`**: The number of epochs to train the model.

For example, to initialize the model with different parameters, modify the following line in the code:

```python
model = VQVAE(input_channels=3, num_embeddings=512, embedding_dim=64)
```

---

## Training the Model

To train the model, you need to specify the dataset and the hyperparameters. The training loop will perform the forward and backward passes, calculate the reconstruction loss, and update the model weights.

```python
# Example training loop snippet:
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for data in train_loader:
        x, _ = data
        x = x.to(device).float()  # Convert to float

        optimizer.zero_grad()

        # Forward pass
        x_recon, z_q = model(x)

        # Compute the loss
        loss = vq_vae_loss(x_recon, x, z_q, x)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
```

Ensure you define the dataset properly, for example, using `torchvision` datasets or custom datasets. The model will optimize the loss function based on the reconstruction error between the original and reconstructed images.

---

## Simulating the Model

Once the model is trained, you can use it for inference (image reconstruction) on a test image.

```python
# Test image reconstruction
test_image, _ = train_dataset[0]
test_image = test_image.unsqueeze(0).to(device)  # Add batch dimension

model.eval()
with torch.no_grad():
    reconstructed_image, _ = model(test_image)

# Visualize results
```

You can visualize the original and reconstructed images using `matplotlib`.

---

## Visualizing Results

To visualize the original and reconstructed images:

```python
import matplotlib.pyplot as plt

# Plot original and reconstructed images
fig, axes = plt.subplots(1, 2)
axes[0].imshow(test_image.squeeze().permute(1, 2, 0).cpu().numpy())  # Original image
axes[0].set_title("Original")

axes[1].imshow(reconstructed_image.squeeze().permute(1, 2, 0).cpu().numpy())  # Reconstructed image
axes[1].set_title("Reconstructed")

plt.show()
```

This will display both the original image and its reconstruction side-by-side.

---

## Additional Considerations

### Early Stopping and Learning Rate Scheduler
- You can implement early stopping and learning rate schedulers to avoid overfitting and optimize the model’s training process. 
- Early stopping is implemented by monitoring the validation loss, and if it doesn't improve for a certain number of epochs (`patience`), the training stops.
- Learning rate schedulers such as `ReduceLROnPlateau` can reduce the learning rate if the validation loss plateaus.

### Gradient Clipping
- **Gradient clipping** helps to prevent exploding gradients, especially when training deep networks. You can clip gradients as shown below:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Output

### Training & Validation Loss Over Epochs
![Screenshot 2025-03-29 053750](https://github.com/user-attachments/assets/64b08f34-b657-4f3f-952b-18903b52db89)

### Original v Reconstructed Image
![Screenshot 2025-03-29 053810](https://github.com/user-attachments/assets/19f44b09-89c0-4115-b322-f015d88f146b)



## Contributing

Contributions to improve the model or extend it to other use cases are welcome! Please fork the repository, create a new branch, and submit a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Input Section for Plot and Image

Below you can add your visualizations directly after training and inference.

- **Training Plot**: Input the training and validation loss plot after each epoch.
- **Reconstructed Image**: Add the original and reconstructed images.

```python
# Input the plot of training/validation losses
# [Insert training loss and validation loss plot here]

# Input the original vs. reconstructed image plot
# [Insert original and reconstructed image here]
```
