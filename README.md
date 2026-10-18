# DIM-IAG-RS 单门控低光图像增强

面向检测和分割的任务导向增强项目。保留 DIM-WRR、单个 IAG 门控及推理阶段 RS 残差调节。

```text
X → DIM-WRR → Y_base
X、Y_base、光照/噪声/边缘代理 → IAG → G
噪声、过曝、欠曝 → RS → S
Y = clamp(X + G*S*(Y_base-X), 0, 1)
```

## 安装与运行

先为设备安装匹配的 torch/torchvision，再运行 `pip install -r requirements.txt`。
原始训练权重未包含，需自行放置；没有权重时不要把随机输出当作增强效果。

```bash
python scripts/enhance_single_gate.py --input_dir /path/to/images --output_dir outputs/enhanced --ckpt weights/dim_iag_rs_best.pth --device cuda:0 --preset medium --save_maps
```

## 历史论文结果

论文配置的 LOD 检测 mAP50–95 是 0.390→0.502，LIS 掩码是 0.303→0.384。原检查点与完整日志缺失，不能保证当前源码封装与历史配置完全一致，本轮未复跑。详见项目档案中的版本说明。

## 来源

保留原源码 MIT License（含原版权声明）；代码继承与本次路径/错误处理整理不构成对上游模块原创性的声明。论文未发表，以个人研究项目呈现。


## 阅读与面试

- [项目档案](docs/project_guide.md)
- [核心问题与递进追问](docs/interview_qa.md)

## 本次整理的验证边界

采用历史材料，不重跑训练。Python 语法检查不能证明模型效果或历史指标已复现。缺失资源明确保留；源码整理和历史实验结果分别记录。
