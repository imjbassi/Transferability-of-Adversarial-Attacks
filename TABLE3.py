import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import foolbox as fb
import numpy as np
from torchvision.models import resnet18, vgg16, mobilenet_v2

# 🔹 Detect Device (Use CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Debug message

# 🔹 Load Pretrained Models
resnet = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
vgg = vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).to(device).eval()
mobilenet = mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1).to(device).eval()

# 🔹 Define Foolbox Models
fmodel_resnet = fb.PyTorchModel(resnet, bounds=(0, 1), device=device)
fmodel_vgg = fb.PyTorchModel(vgg, bounds=(0, 1), device=device)
fmodel_mobilenet = fb.PyTorchModel(mobilenet, bounds=(0, 1), device=device)

# 🔹 Load CIFAR-10 Dataset (Test Set Only)
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# 🔹 Define FGSM Attack (Fix epsilons Parameter)
attack = fb.attacks.FGSM()

# 🔹 Function to Evaluate Transferability
def evaluate_transferability(source_model, target_model, data_loader):
    success = 0
    total = 0
    
    for img, label in data_loader:
        img, label = img.to(device), label.to(device)  # Ensure tensors are on the correct device
        
        # Generate adversarial example (Fix 'eps' → 'epsilons')
        raw_advs, clipped_advs, success_batch = attack(source_model, img, label, epsilons=0.03)
        
        # Test adversarial example on target model
        preds = torch.argmax(target_model(clipped_advs), dim=1)
        success += (preds != label).sum().item()
        total += label.size(0)
    
    return 100 * success / total

# 🔹 Evaluate Transferability Between Models
results = {
    "ResNet18 → VGG16": evaluate_transferability(fmodel_resnet, vgg, testloader),
    "ResNet18 → MobileNetV2": evaluate_transferability(fmodel_resnet, mobilenet, testloader),
    "VGG16 → ResNet18": evaluate_transferability(fmodel_vgg, resnet, testloader),
    "VGG16 → MobileNetV2": evaluate_transferability(fmodel_vgg, mobilenet, testloader),
    "MobileNetV2 → ResNet18": evaluate_transferability(fmodel_mobilenet, resnet, testloader),
    "MobileNetV2 → VGG16": evaluate_transferability(fmodel_mobilenet, vgg, testloader),
}

# 🔹 Print Results
for key, value in results.items():
    print(f"{key}: {value:.2f}% transfer success rate")
