# DCGAN: MNIST Digit Generator

This notebook trains a Deep Convolutional GAN (DCGAN) to generate handwritten digit images similar to MNIST.  
It outputs sample images per epoch and a final animation (`dcgan.gif`) showing training progress.

```python
# --- Dependencies ---
# tensorflow 2.4.1, numpy, matplotlib, scikit-image, imageio

# --- Constants ---
BUFFER_SIZE = 60000
BATCH_SIZE = 2048
NOISE_DIM = 100
EPOCHS = 250

# --- Generator ---
# Takes 100-dim noise → Dense → Reshape → Upsamples to 32×32×1 via Conv2DTranspose
def make_generator_model():
    ...

# --- Discriminator ---
# Takes 32×32×1 image → Conv2D → Flatten → Outputs real/fake score
def make_discriminator_model():
    ...

# --- Losses ---
# Binary crossentropy on real vs fake outputs
def gen_loss(fake): ...
def disc_loss(real, fake): ...

# --- Training ---
# For each batch: generate fake → get discriminator output → compute & apply gradients
@tf.function
def train_step(images):
    ...

# --- Output ---
# After each epoch, generated images are saved as PNGs.
# At the end, all PNGs are combined into dcgan.gif:
with imageio.get_writer('dcgan.gif', mode='I') as writer:
    for f in sorted(glob.glob('image_at_epoch_*.png')):
        writer.append_data(imageio.imread(f))

This project demonstrates how GANs can learn to generate realistic digit images from noise.
Use it for dataset augmentation, educational visualization, or as a GAN boilerplate.
