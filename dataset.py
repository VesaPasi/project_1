import cv2
from torch.utils.data import Dataset


def read_xray(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    xray = xray.astype('float32') / 255.0  # Normalize to [0, 1]
    xray = xray.reshape((1, *xray.shape))  # Add channel dimension
    return xray

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype('float32')
    mask = mask.reshape((1, *mask.shape))  # Add channel dimension
    return mask

class Knee_Dataset(Dataset):
    def __init__(self, df):
        self.df = df
       

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = read_xray(self.df["xrays"].iloc[idx])
        mask = read_mask(self.df["masks"].iloc[idx])

        res = {
            "image": image,
            "mask": mask
        }

        return res