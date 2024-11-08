import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from modelling import Generator

latent_dim = 128
image_size = (128, 128)

def load_generator(model_path, latent_dim, image_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim=latent_dim, image_size=image_size).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval() 
    return generator

def generate_image(generator, latent_dim, num_images=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn(num_images, latent_dim, device=device)
    with torch.no_grad():
        generated_imgs = generator(z)
    return generated_imgs

def display_images(images, title="Generated Images"):
    images = (images * 0.5 + 0.5).clamp(0, 1)
    grid_img = make_grid(images, nrow=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img.permute(1, 2, 0).cpu())
    plt.title(title)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    model_path = "generator_epoch_0.pth"

    generator = load_generator(model_path, latent_dim, image_size)

    num_images = 16 
    generated_images = generate_image(generator, latent_dim, num_images=num_images)

    display_images(generated_images)
