"""
Self-contained Minimal implementation of VQ-GAN model.
Paper: https://arxiv.org/abs/2012.09841
Reference: https://compvis.github.io/taming-transformers
Copyright: Do whatever you want. Don't ask me.

"""

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SimpleDataset(Dataset):
    def __init__(self, image_list_file, size=256):
        super().__init__()
        with open(image_list_file, "r") as f:
            self.list_img_paths = f.read().splitlines()
        self._length = len(self.list_img_paths)
        self.preprocess_image = transforms.Compose([
            transforms.Resize(size=size), # Resize min-size to 256 and maintains aspect-ratio. Original implementation has max-size to 256
            transforms.RandomCrop(256),   # Crop 256x256 to have 16x16 during encoding with self-attention
            transforms.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1))  # Convert to tensor without normalization
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        img_path = self.list_img_paths[index]
        image = Image.open(img_path).convert("RGB")
        image = self.preprocess_image(image)
        image = (image / 127.5) - 1.0 # manual normalisation of image in [-1.0, 1.0]
        return image
        

if __name__ == "__main__":
    """
    Testing dataset and dataloader.

    """
    dataset = SimpleDataset(
        image_list_file = "/path/to/open-images-dataset/test.txt",
        size=256
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, num_workers=8, pin_memory=True, drop_last=True)

    for i_data, data in enumerate(loader):
        print(f"Index: {i_data:03d} | Data_Shape: {data.shape} | min_val: {data.min():.3f} | max_val: {data.max():.3f}")