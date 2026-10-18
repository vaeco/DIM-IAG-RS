# 方法与代码结构

## 基础增强

`dim_iag_rs/dim.py` 实现DIM，卷积下采样提取特征，经过深层表示与上采样融合得到增强输出。强度映射 `2x-x²` 作用于中间特征，不保证任意输入特征都处于[0,1]。

`dim_wrr.py` 将原图与DIM输出做Haar分解，两组RGB系数拼接为24通道，预测小波残差后逆变换为RGB修正。`base_alpha` 控制修正强度：

```text
Y_base = clamp(Y_dim + base_alpha * (Y_wrr - Y_dim), 0, 1)
```

## 单门控IAG

`dim_wrr_iag.py` 预测照明与噪声信息，再融合16通道输入输出 `[B,1,H,W]` 门控图：原图、基础增强、反射率代理各3通道，光照、噪声、亮度、边缘、残差幅度、欠曝和过曝各1通道。

反射率代理采用 `clamp(X/(I+eps),0,1)`，仅作为辅助表示。模型本身输出 `clamp(X+G*(Y_base-X),0,1)`，单通道G按RGB广播。

## 推理风险调节RS

RS位于 `scripts/enhance_single_gate.py`，读取门控与中间结果，计算：

```text
S = clamp(1 - w_noise*N - w_over*O + w_under*U, s_min, s_max)
Y = clamp(X + G*S*(Y_base-X), 0, 1)
```

噪声N和过曝O降低候选残差，欠曝U保留更多增强。S在部分preset允许略大于1，因此它是调节图而非概率。该步骤无需下游标签，也不增加训练监督损失。

## 入口与配置

按 `DIM.forward → DIM_WRR.forward → DIM_WRR_IAG.forward → enhance_single_gate.py` 跟踪数据。推理支持mild、medium、strong和custom设置；具体权重以脚本PRESETS为准。仅调用模型forward不会自动执行RS，应使用对应推理入口。

## 评测要求

原图与增强图使用相同下游权重、数据划分、尺寸与评价规则。LOD使用检测指标，LIS区分框与掩码指标，分类与语义分割工具需要各自依赖和检查点。数据和权重不随仓库提供。
