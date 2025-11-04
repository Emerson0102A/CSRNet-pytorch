import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mindspore as ms
import numpy as np
from PIL import Image
from matplotlib import cm
from mindspore import Tensor, context, ops
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from ms_model import CSRNet  # 需提供 MindSpore 版 CSRNet 定义

#python infer_single_mindspore.py


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "csrnet_from_torch.ckpt"
DEFAULT_IMAGE = BASE_DIR / "IMG_11.jpg"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MindSpore CSRNet inference")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL, help="checkpoint (.ckpt) path")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE, help="input image path")
    parser.add_argument(
        "--device-target",
        type=str,
        default="CPU",
        choices=["GPU", "CPU", "Ascend"],
        help="MindSpore device target",
    )
    parser.add_argument("--graph-mode", action="store_true", help="use GRAPH_MODE instead of default PYNATIVE_MODE")
    return parser.parse_args()


def configure_runtime(device_target: str, graph_mode: bool) -> None:
    mode = context.GRAPH_MODE if graph_mode else context.PYNATIVE_MODE
    context.set_context(mode=mode, device_target=device_target.upper())


def preprocess_image(image_path: Path) -> tuple[np.ndarray, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    img_np = np.asarray(image).astype(np.float32) / 255.0
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np, image


def load_model(model_path: Path) -> CSRNet:
    net = CSRNet(load_weights=True)
    param_dict = load_checkpoint(str(model_path))
    load_param_into_net(net, param_dict)
    net.set_train(False)
    return net


def run_inference(net: CSRNet, image_tensor: Tensor) -> Tensor:
    density_map = net(image_tensor)
    return density_map


def visualize_results(image: Image.Image, density_map: np.ndarray, predicted_count: float) -> None:
    density_vis = cv2.resize(density_map, (image.width, image.height), interpolation=cv2.INTER_CUBIC)
    vis_sum = density_vis.sum()
    if vis_sum > 0:
        density_vis *= density_map.sum() / vis_sum
    density_norm = density_vis / (density_vis.max() + 1e-12)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Input Image")

    overlay_ax = plt.subplot(1, 2, 2)
    overlay_ax.imshow(image)
    heatmap = overlay_ax.imshow(density_norm, cmap=cm.jet, alpha=0.6)
    plt.axis("off")
    plt.title(f"Density Overlay (estimate={predicted_count:.1f})")
    plt.colorbar(heatmap, ax=overlay_ax, fraction=0.046, pad=0.04, label="Density")
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    configure_runtime(args.device_target, args.graph_mode)

    if not args.model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")
    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    image_np, image = preprocess_image(args.image_path)
    image_tensor = Tensor(image_np[np.newaxis, ...], dtype=ms.float32)

    net = load_model(args.model_path)
    density_map = run_inference(net, image_tensor)

    predicted_count = float(ops.ReduceSum()(density_map).asnumpy())
    print(f"Predicted crowd count: {predicted_count:.2f}")

    density_np = density_map.asnumpy().squeeze()
    visualize_results(image, density_np, predicted_count)


if __name__ == "__main__":
    main()
