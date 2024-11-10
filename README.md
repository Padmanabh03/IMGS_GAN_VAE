# VAE and GAN Applications on MNIST and CIFAR-10

This repository contains implementations of Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN) applied to the MNIST and CIFAR-10 datasets. The project covers various applications of generative models, including image generation, latent space exploration, anomaly detection, and visualization.

## Project Overview

1. **GAN Implementation on MNIST**
   - Implemented a simple GAN model to generate images from the MNIST dataset using a fully connected generator and discriminator. Training snapshots capture the progression of generated images over multiple epochs, and loss curves illustrate training stability and generator-discriminator dynamics.

2. **DCGAN Implementation on CIFAR-10**
   - Built a Deep Convolutional GAN (DCGAN) to generate images from the CIFAR-10 dataset. Best practices such as using strided convolutions, batch normalization, and appropriate activation functions (ReLU in the generator and Leaky ReLU in the discriminator) were applied to improve training stability and image quality. The generated images are compared with real CIFAR-10 images.

3. **Latent Space Interpolation in GANs**
   - Performed latent space interpolation in the GAN by selecting two distinct latent vectors and generating intermediate images through linear interpolation. This demonstrates the model’s ability to generate smooth transitions between different images, highlighting the continuity of the learned latent space.

4. **VAE Implementation on MNIST**
   - Developed a Variational Autoencoder (VAE) to reconstruct images from the MNIST dataset and generate new samples by sampling from the learned latent space. The model's performance was evaluated with the ELBO loss and KL-divergence, providing insight into the trade-offs between reconstruction fidelity and latent space regularization.

5. **VAE for Anomaly Detection**
   - Applied the VAE for anomaly detection by evaluating the reconstruction error on the MNIST dataset. Anomalous images, such as noisy or corrupted digits, were expected to have higher reconstruction errors. A threshold was set to classify images as either "normal" or "anomalous," and the model was tested on a mix of normal and anomalous images.

   - **Anomaly Detection Results**:
     - **Threshold**: Set a threshold for classification based on reconstruction error.
     - **False Positives**: Number of normal images classified as anomalous.
     - **True Positives**: Number of anomalous images correctly classified as anomalous.

6. **VAE Latent Space Visualization**
   - Visualized the VAE’s latent space by using t-SNE to project the latent vectors into 2D, with points color-coded by digit class (0-9). This visualization shows how well the model clusters different digit classes, indicating that the VAE has learned a structured latent space with clear groupings for similar digits.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- scikit-learn (for t-SNE visualization)





