import os
import kaggle

def download_celeba_data(kaggle_dataset="jessicali9530/celeba-dataset", download_path="data/celeba"):
    data_dir = os.path.join(download_path, "img_align_celeba")
    if not os.path.exists(data_dir):
        print("Dataset not found. Downloading from Kaggle...")
        os.makedirs(download_path, exist_ok=True)
        kaggle.api.dataset_download_files(kaggle_dataset, path=download_path, unzip=True)
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already exists. Skipping download.")

if __name__ == "__main__":
    download_celeba_data()
