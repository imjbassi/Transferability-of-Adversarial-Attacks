import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from foolbox import PyTorchModel, attacks

# 🔹 Detect Device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🔹 Load CIFAR-10 Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
image, label = dataset[0]  # Get a single image

# 🔹 Convert image to tensor & move to device
image = image.unsqueeze(0).to(device)  # Add batch dimension & move to GPU/CPU
label = torch.tensor([label], device=device)  # Ensure label is on the same device

# 🔹 Load Pretrained Model (ResNet18)
model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True).to(device).eval()

# 🔹 Setup Foolbox Model
fmodel = PyTorchModel(model, bounds=(-1, 1), device=device)

# 🔹 Define FGSM Attack
attack = attacks.LinfFastGradientAttack()  # Correct attack usage
epsilon = 0.03  # Perturbation magnitude

# 🔹 Generate Adversarial Example
adversarial = attack.run(fmodel, image, label, epsilon=epsilon)

# 🔹 Convert Tensors to Images for Visualization
def denormalize(img):
    """ Reverses normalization for visualization. """
    return (img * 0.5) + 0.5  # Undo normalization

# 🔹 Plot Original & Adversarial Images
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(denormalize(image.squeeze()).permute(1, 2, 0).cpu().numpy())  # Original image
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(denormalize(adversarial.squeeze()).permute(1, 2, 0).cpu().numpy())  # Adversarial image
axes[1].set_title("Adversarial Image")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("adversarial_example.png")  # Save adversarial image
plt.show()
