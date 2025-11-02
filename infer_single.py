import torch
from PIL import Image
from torchvision import transforms

from model import CSRNet  # 同 repo 内的模型定义

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

MODEL_PATH = "0model_best.pth.tar"  # 或其他 checkpoint
IMAGE_PATH = "/root/autodl-tmp/CSRNet-pytorch/IMG_4.jpg"

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
density_np = density_map.squeeze().cpu().numpy()


# 归一化方便显示
density_norm = density_np / (density_np.max() + 1e-12)

plt.figure(figsize=(12, 5))

# 原图
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis("off")
plt.title("Input Image")

# 热力图（用 jet / plasma 等 colormap）
plt.subplot(1, 2, 2)
plt.imshow(density_norm, cmap=cm.jet)
plt.colorbar(label="Density")
plt.axis("off")
plt.title(f"Density Map (count={predicted_count:.1f})")

plt.tight_layout()
plt.show()