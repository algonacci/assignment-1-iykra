import os
from dotenv import load_dotenv

load_dotenv()
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")
import torch
import torch.nn as nn
import torch.optim as optim
from data_ingestion import download_celeba_data
from etl import load_celeba_data
from modelling import Generator, Discriminator

# Hyperparameters
latent_dim = 128
batch_size = 128
image_size = (128, 128)
n_epochs = 100
lr = 0.0001


def main():
    # Step 1: Data Ingestion
    download_celeba_data()

    # Step 2: Data Loading
    data_dir = "data/celeba/img_align_celeba"
    dataloader = load_celeba_data(
        data_dir=data_dir, batch_size=batch_size, image_size=image_size
    )

    # Step 3: Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim=latent_dim, image_size=image_size).to(device)
    discriminator = Discriminator(image_size=image_size).to(device)

    # Step 4: Loss and Optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training Loop
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # Move data to device
            real_imgs = imgs.to(device)
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            print(
                f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
            )

        if epoch % 10 == 0:
            torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch}.pth")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0)} bytes")
        print(f"Memory cached: {torch.cuda.memory_reserved(0)} bytes")
    else:
        print("No GPU detected. Using CPU.")

    main()
