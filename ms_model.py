"""MindSpore implementation of CSRNet."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Union

import mindspore as ms
from mindspore import Tensor, nn
from mindspore.common.initializer import Constant, Normal, initializer

# ✅ 使用 MindCV 的 VGG16
try:
    from mindcv.models import vgg16 as mindcv_vgg16
except Exception as e:
    raise RuntimeError(
        "未找到 mindcv，请先 `pip install -U mindcv`。"
    ) from e


LayerCfg = Sequence[Union[int, str]]


def make_layers(
    cfg: LayerCfg,
    in_channels: int = 3,
    batch_norm: bool = False,
    dilation: bool = False,
) -> nn.SequentialCell:
    d_rate = 2 if dilation else 1
    layers: List[nn.Cell] = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(v),
            kernel_size=3,
            stride=1,
            pad_mode="pad",
            padding=d_rate,
            dilation=d_rate,
            has_bias=True,
        )
        if batch_norm:
            layers.extend([conv, nn.BatchNorm2d(int(v)), nn.ReLU()])
        else:
            layers.extend([conv, nn.ReLU()])
        in_channels = int(v)
    return nn.SequentialCell(layers)


class CSRNet(nn.Cell):
    def __init__(self, load_weights: bool = False) -> None:
        super().__init__()
        self.seen = 0
        # CSRNet 前端：VGG16 到 conv4_3（不含 pool4/conv5）
        self.frontend_feat: LayerCfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        self.backend_feat: LayerCfg = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
            stride=1,
            pad_mode="pad",
            padding=0,
            has_bias=True,
        )
        self._initialize_weights()
        if not load_weights:
            self._load_vgg_frontend()

    def construct(self, x: Tensor) -> Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self) -> None:
        normal_init = Normal(sigma=0.01, mean=0.0)
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(initializer(normal_init, cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer(Constant(0.0), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(initializer(Constant(1.0), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer(Constant(0.0), cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(normal_init, cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer(Constant(0.0), cell.bias.shape, cell.bias.dtype))

    def _load_vgg_frontend(self) -> None:
        """
        用 MindCV 的 VGG16 预训练参数初始化前端特征提取层。
        兼容 vgg.features 是否存在的两种情况。
        """
        try:
            vgg = mindcv_vgg16(pretrained=True)  # 需要网络下载或本地缓存
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "加载 VGG16 预训练权重失败，请确认已连接网络或已缓存权重（可先尝试 `pip install -U mindcv`）。"
            ) from exc

        # 取出 VGG 的参数序列（优先 features）
        if hasattr(vgg, "features"):
            vgg_params: Iterable[ms.Parameter] = vgg.features.get_parameters()
        else:
            vgg_params = vgg.get_parameters()

        frontend_params: Iterable[ms.Parameter] = self.frontend.get_parameters()

        # 逐个参数对齐拷贝（形状不一致则跳过）
        for own_param, vgg_param in zip(frontend_params, vgg_params):
            if getattr(own_param, "shape", None) == getattr(vgg_param, "shape", None):
                own_param.set_data(vgg_param.data)
