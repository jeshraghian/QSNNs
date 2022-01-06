from torchvision import datasets, transforms

# from snntorch import utils


def load_data(config):
    data_dir = config["data_dir"]

    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ]
    )

    trainset = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform
    )
    testset = datasets.FashionMNIST(
        data_dir, train=False, download=True, transform=transform
    )

    # enable this following to reduce the dataset by 100x for speed

    # utils.data_subset(trainset, 100)
    # utils.data_subset(testset, 100)

    return trainset, testset
