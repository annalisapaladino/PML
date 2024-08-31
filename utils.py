import torch
import torchvision
import PIL



# Custom dataset
class FacesDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor(),
        ])
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.images[idx]).convert('RGB')
        x = self.transform(image)
        y = x
        return x, y