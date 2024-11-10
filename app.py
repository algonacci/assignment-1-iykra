import os
from flask import Flask, jsonify, send_from_directory, request
import torch
import os
from torchvision.utils import save_image
from modelling import Generator
import gdown

app = Flask(__name__)
app.config["STATIC_FOLDER"] = "static"

latent_dim = 128
image_size = (128, 128)
model_url = "https://drive.google.com/uc?id=1UqjsDF4sIcyFpXJpu8tSzaKt5VG3ogLa"

model_path = os.path.join(app.config["STATIC_FOLDER"], "generator_epoch_0.pth")


def download_model():
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        gdown.download(model_url, model_path, quiet=False)
        print("Model downloaded successfully.")


def load_generator(model_path, latent_dim, image_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim=latent_dim, image_size=image_size).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    return generator


@app.route("/")
def index():
    return jsonify(
        {
            "status": {
                "code": 200,
                "message": "Success fetching the API",
            },
            "data": {"endpoint": f"{request.host_url}generate_image"},
        }
    ), 200


@app.route("/generate_image", methods=["GET"])
def generate_image_endpoint():
    download_model()
    generator = load_generator(model_path, latent_dim, image_size)

    z = torch.randn(
        16,
        latent_dim,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    with torch.no_grad():
        generated_imgs = generator(z)

    image_filename = "generated_image_grid.png"
    image_path = os.path.join(app.config["STATIC_FOLDER"], image_filename)
    save_image(generated_imgs * 0.5 + 0.5, image_path, nrow=4)

    return send_from_directory(app.config["STATIC_FOLDER"], image_filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
