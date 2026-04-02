import os
import torch
from PIL import Image
from torchvision import transforms
import pyiqa
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hr_path = "./results_reallr/sample00"
gt_path = None

metrics = {
    # Full-reference
    "psnr": pyiqa.create_metric('psnr').to(device),
    "ssim": pyiqa.create_metric('ssim').to(device),
    "lpips": pyiqa.create_metric('lpips').to(device),

    # No-reference
    "niqe": pyiqa.create_metric('niqe').to(device),
    "maniqa": pyiqa.create_metric('maniqa').to(device),
    "musiq": pyiqa.create_metric('musiq').to(device),
}

transform = transforms.ToTensor()

hr_files = sorted(os.listdir(hr_path))

results = {k: [] for k in metrics.keys()}

use_gt = gt_path is not None and os.path.exists(gt_path)
print(f"Use GT: {use_gt}")

for fname in tqdm(hr_files):
    hr_img_path = os.path.join(hr_path, fname)

    # load HR
    hr = Image.open(hr_img_path).convert("RGB")
    hr = transform(hr).unsqueeze(0).to(device)

    gt = None
    if use_gt:
        gt_img_path = os.path.join(gt_path, fname)
        if os.path.exists(gt_img_path):
            gt = Image.open(gt_img_path).convert("RGB")
            gt = transform(gt).unsqueeze(0).to(device)
        else:
            gt = None

    with torch.no_grad():
        # ---- Full-reference ----
        if gt is not None:
            results["psnr"].append(metrics["psnr"](hr, gt).item())
            results["ssim"].append(metrics["ssim"](hr, gt).item())
            results["lpips"].append(metrics["lpips"](hr, gt).item())

        # ---- No-reference ----
        results["niqe"].append(metrics["niqe"](hr).item())
        results["maniqa"].append(metrics["maniqa"](hr).item())
        results["musiq"].append(metrics["musiq"](hr).item())

print("\n==== Average Results ====")
for k, v in results.items():
    if len(v) > 0:
        print(f"{k.upper():6s}: {sum(v)/len(v):.4f}")
    else:
        print(f"{k.upper():6s}: N/A (no GT)")