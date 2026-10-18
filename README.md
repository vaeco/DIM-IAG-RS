# DIM-IAG-RS

面向下游视觉任务的单门控低光图像增强实现。结合强度映射、Haar小波残差、光照与噪声感知门控，以及推理阶段的风险残差调节，在提亮、结构保留和噪声抑制之间控制增强幅度。

## 方法

```text
X → DIM-WRR → Y_base
X、Y_base与光照/噪声/边缘信息 → IAG → G
噪声、过曝、欠曝风险 → RS → S
Y = clamp(X + G * S * (Y_base - X), 0, 1)
```

`G` 是学习得到的单通道空间门控，`S` 是推理脚本计算的风险调节图。实现细节见 [方法与代码结构](docs/project_guide.md)。

## 安装与推理

先安装适配设备的PyTorch和torchvision，然后执行：

```bash
python -m pip install -r requirements.txt
python scripts/enhance_single_gate.py --input_dir /path/to/images --output_dir outputs/enhanced --ckpt weights/dim_iag_rs_best.pth --device cuda:0 --preset medium --save_maps
```

原训练检查点未随仓库分发，运行需要兼容的单门控权重。使用 `--device cpu` 可显式选择CPU。`--save_maps` 保存门控与风险图，参数说明见 `--help`。

## 实验结果

| 论文配置 | LOD 检测mAP50-95 | LIS 掩码mAP50-95 |
|---|---:|---:|
| 未增强基线 | 0.390 | 0.303 |
| DIM-IAG-RS | 0.502 | 0.384 |

对应绝对差为11.2和8.1个百分点。结果来自研究稿的实验配置；原始权重与完整日志未分发，当前代码与该配置尚未完成一致性验证。详细指标与适用范围见 [实验说明](docs/experiments.md)。

## 目录

- `dim_iag_rs/` 基础增强、小波残差和单门控模型
- `scripts/enhance_single_gate.py` 推理与风险调节
- `scripts/validate_release.py` 检查点严格加载检查
- `configs/` 数据集与评测配置示例
- `tools/eval/` 下游评价工具，见 [评测说明](tools/eval/README.md)
- [权重说明](weights/README.md) · [数据说明](datasets/README.md)

## 限制

增强收益依赖数据域、固定下游模型及评测设置。风险图属于估计而非物理真值，强抑制可能保留欠曝，强增强可能放大噪声。检查点加载或语法检查不等于下游指标复现。

## 许可证

使用 [MIT License](LICENSE)，保留原始版权声明与上游实现的授权信息。
