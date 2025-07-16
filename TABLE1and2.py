import torch
import torchvision
import torchvision.transforms as transforms
import foolbox as fb
import numpy as np
from torchvision.models import resnet18, vgg16, mobilenet_v2

# 🔹 Detect Device (Use CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🔹 Load Pretrained Models
resnet = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
vgg = vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).to(device).eval()
mobilenet = mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1).to(device).eval()

# 🔹 Define Foolbox Models
fmodel_resnet = fb.PyTorchModel(resnet, bounds=(0, 1), device=device)
fmodel_vgg = fb.PyTorchModel(vgg, bounds=(0, 1), device=device)
fmodel_mobilenet = fb.PyTorchModel(mobilenet, bounds=(0, 1), device=device)

# 🔹 Load CIFAR-10 Dataset (Test Set Only, Limited to 1000 images)
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
subset_indices = np.random.choice(len(testset), 1000, replace=False)  # Select only 1000 samples
subset = torch.utils.data.Subset(testset, subset_indices)
testloader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)  # Increased batch size

# 🔹 Define Attack Methods (Optimized PGD & CW)
attacks = {
    "PGD": fb.attacks.LinfPGD(rel_stepsize=0.01, steps=10),  # Reduced PGD steps to 10
    "CW": fb.attacks.L2CarliniWagnerAttack(binary_search_steps=5, steps=10),  # Reduced CW iterations
}

# 🔹 Function to Evaluate Transferability
def evaluate_transferability(source_model, target_model, data_loader, attack, eps=0.03):
    success = 0
    total = 0
    
    for img, label in data_loader:
        img, label = img.to(device), label.to(device)

        # Generate adversarial example on the source model
        _, clipped_advs, success_batch = attack(source_model, img, label, epsilons=eps)

        # Test on target model
        preds = torch.argmax(target_model(clipped_advs), dim=1)
        success += (preds != label).sum().item()
        total += label.size(0)
    
    return 100 * success / total

# 🔹 Run Transferability Evaluations for PGD & CW (Optimized)
transferability_results = {}

for attack_name, attack in attacks.items():
    transferability_results[attack_name] = {
        "ResNet18 → VGG16": evaluate_transferability(fmodel_resnet, vgg, testloader, attack),
        "ResNet18 → MobileNetV2": evaluate_transferability(fmodel_resnet, mobilenet, testloader, attack),
        "VGG16 → ResNet18": evaluate_transferability(fmodel_vgg, resnet, testloader, attack),
        "VGG16 → MobileNetV2": evaluate_transferability(fmodel_vgg, mobilenet, testloader, attack),
        "MobileNetV2 → ResNet18": evaluate_transferability(fmodel_mobilenet, resnet, testloader, attack),
        "MobileNetV2 → VGG16": evaluate_transferability(fmodel_mobilenet, vgg, testloader, attack),
    }

# 🔹 Print Results (Table 1 & Table 2)
print("\nTransferability of PGD and CW Attacks:")
for attack, results in transferability_results.items():
    print(f"\n{attack}:")
    for key, value in results.items():
        print(f"  {key}: {value:.2f}%")
