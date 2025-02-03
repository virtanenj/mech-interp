import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from torchinfo import summary
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def load_pretrained_resnet():
    # Load ResNet18
    resnet_model = models.resnet18(weights="IMAGENET1K_V1") # weights = "IMAGENET1K_V1" or "DEFAULT"
    resnet_model.eval()
    return resnet_model

def prepare_image_transform():
    # Standard ImageNet transforms
    return transforms.Compose([
        transforms.Resize(224), # Resize to 224x224 (ResNet input size)
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_imagenet_data(batch_size=32):
    transform = prepare_image_transform()
    # ImageNetv2 test set (smaller than full ImageNet)
    testset = datasets.CIFAR100(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    return testloader

def show_image(img_tensor):
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = img_tensor * std + mean
    
    # Convert to PIL image
    img_np = img_tensor.numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    plt.axis("off")
    plt.show()

def register_hooks(model):
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hooks
    model.layer1.register_forward_hook(get_activation('layer1'))
    model.layer2.register_forward_hook(get_activation('layer2'))
    model.layer3.register_forward_hook(get_activation('layer3'))
    model.layer4.register_forward_hook(get_activation('layer4'))

    return activation

def visualize(): pass

if __name__ == "__main__":
    model = load_pretrained_resnet()
    # print("\nModel Summary:")
    # print(summary(model, input_size=(1, 3, 224, 224)))
    # print(model)

    testloader = get_imagenet_data()

    images, labels = next(iter(testloader))
    with torch.no_grad():
        output = model(images)
        _, predicted = torch.max(output, 1)
    
    print(f"Batch shape: {images.shape}")
    print(f"Number of classes: {len(testloader.dataset.classes)}")

    print("Showing image...")
    print(f"Label: {labels[0]}")
    show_image(images[0])


    # activation = register_hooks(model)

    # transform = prepare_image_transform()
    # img = Image.open("data/elephant.jpg")
    # input_tensor = transform(img).unsqueeze(0)

    # with torch.no_grad():
    #     output = model(input_tensor)
    
    # for name, act in activation.items():
    #     print(f"{name} shape: {act.shape}")



    

    