from os.path import join
from torchvision import transforms
from datasets import DatasetFromFolder, DatasetFromFolder_test
from torch.utils.data import DataLoader


def transform():
    return transforms.Compose([
        # ColorJitter(hue=0.3, brightness=0.3, saturation=0.3),
        # RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def get_training_set(data_dir, ref_dir):

    train_set = DatasetFromFolder(data_dir, ref_dir, fineSize=256)

    # Pytorch train and test sets
    # tensor_dataset = torch.utils.data.TensorDataset(train_set)

    return train_set

def get_testing_set(data_dir):

    train_set = DatasetFromFolder_test(data_dir, fineSize=256)

    # Pytorch train and test sets
    # tensor_dataset = torch.utils.data.TensorDataset(train_set)

    return train_set





