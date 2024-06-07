import torchvision.transforms as transforms

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.grouper import CombinatorialGrouper


# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="iwildcam", download=True)
print(dataset)
print(type(dataset))
print('dataset size = ', len(dataset))

# Get the training set
train_data = dataset.get_subset(
    "train",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)

# Prepare the standard data loader
train_loader = get_train_loader("standard", train_data, batch_size=16)


# Train loop
for labeled_batch in train_loader:
    x, y, metadata = labeled_batch
    print(metadata)
    break


grouper = CombinatorialGrouper(dataset, ['location'])
for x, y_true, metadata in train_loader:
    z = grouper.metadata_to_group(metadata)
    print(z)
    break
