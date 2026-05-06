import os
import torch
import matplotlib.pyplot as plt

from monai.transforms import LoadImage, EnsureChannelFirst
from surgical_copilot.bench.perturbation import PerturbationPipelines 
from surgical_copilot.HemoDataset import HemosetDataSet

def load_image(path):
    loader = LoadImage(image_only=True)
    img = loader(path)
    img = EnsureChannelFirst()(img)
    return torch.tensor(img)

def normalize(img):
    img = img.clone()
    img -= img.min()
    img /= (img.max() + 1e-8)
    return img


def visualize(image, save_dir="outputs", max_scenarios=6):
    os.makedirs(save_dir, exist_ok=True)

    scenarios = PerturbationPipelines.get_eval_scenarios()

    image = image.unsqueeze(0)  # add batch dim

    for i, (name, pipeline) in enumerate(scenarios.items()):

        data = {"image": image.clone()}
        pert = pipeline(data)["image"]

        original = image[0, 0].cpu()
        perturbed = pert[0, 0].cpu()
        diff = torch.abs(perturbed - original)

        original = normalize(original)
        perturbed = normalize(perturbed)
        diff = normalize(diff)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(original, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title(f"Perturbed\n{name}")
        plt.imshow(perturbed, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Difference")
        plt.imshow(diff, cmap="hot")
        plt.axis("off")

        save_path = os.path.join(save_dir, f"{name}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"[✓] Salvato: {save_path}")


if __name__ == "__main__":
    
    dataset = HemosetDataSet(root_dir="data/raw")
    sample = dataset.get_sample(patient_id="pig3")
    visualize(sample["image"])