import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

from model import CSRNet  # 同 repo 内的模型定义

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

##python infer_single.py

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "0model_best.pth.tar"  # 或其他 checkpoint
IMAGE_PATH = BASE_DIR / "IMG_5.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 1. 加载模型
checkpoint = torch.load(MODEL_PATH, map_location=device)
model = CSRNet().to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# 2. 读入图片并做预处理
img = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# 3. 前向推理
with torch.no_grad():
    density_map = model(img_tensor)

predicted_count = density_map.sum().item()
print(f"Predicted crowd count: {predicted_count:.2f}")

# 如果需要可视化密度图：
# 将密度图插值到输入分辨率，便于视觉上更平滑
density_map_vis = F.interpolate(
    density_map,
    size=img_tensor.shape[-2:],
    mode="bilinear",
    align_corners=False,
)

vis_total = density_map_vis.sum()
if vis_total > 0:
    density_map_vis *= density_map.sum() / vis_total

density_np = density_map_vis.squeeze().cpu().numpy()


# 归一化方便显示
density_norm = density_np / (density_np.max() + 1e-12)

plt.figure(figsize=(12, 5))

# 原图
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis("off")
plt.title("Input Image")

# 热力图叠加在原图上，便于查看高密度区域
overlay_ax = plt.subplot(1, 2, 2)
overlay_ax.imshow(img)
heatmap = overlay_ax.imshow(density_norm, cmap=cm.jet, alpha=0.6)
plt.axis("off")
plt.title(f"Density Overlay (estimate={predicted_count:.1f})")
plt.colorbar(heatmap, ax=overlay_ax, fraction=0.046, pad=0.04, label="Density")

plt.tight_layout()
plt.show()