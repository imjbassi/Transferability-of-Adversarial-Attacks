import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Load pretrained models
models_dict = {
    "ResNet18": models.resnet18(pretrained=True).to(device).eval(),
    "VGG16": models.vgg16(pretrained=True).to(device).eval(),
    "MobileNetV2": models.mobilenet_v2(pretrained=True).to(device).eval(),
}

# FGSM attack function
def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    
    images.requires_grad = True
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    
    perturbed_images = images + epsilon * images.grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1)  # Ensure valid pixel range
    
    return perturbed_images

# Evaluate FGSM transferability
epsilon = 8 / 255  # Perturbation size

results = {}

for source_model_name, source_model in models_dict.items():
    source_success_rates = {}
    
    print(f"\nGenerating FGSM adversarial examples from {source_model_name}...")
    
    total_samples = 0
    correct_transfer = {target_model_name: 0 for target_model_name in models_dict.keys()}
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = fgsm_attack(source_model, images, labels, epsilon)

        for target_model_name, target_model in models_dict.items():
            outputs = target_model(adv_images)
            _, predicted = torch.max(outputs, 1)
            correct_transfer[target_model_name] += (predicted == labels).sum().item()
        
        total_samples += labels.size(0)

    # Store transfer success rates
    for target_model_name in models_dict.keys():
        success_rate = 100 * (1 - correct_transfer[target_model_name] / total_samples)
        source_success_rates[target_model_name] = success_rate
        print(f"  {source_model_name} → {target_model_name}: {success_rate:.2f}% transfer success rate")
    
    results[source_model_name] = source_success_rates

# Print final FGSM results
print("\nTransferability of FGSM Attacks:")
for source, targets in results.items():
    for target, success in targets.items():
        print(f"  {source} → {target}: {success:.2f}%")

