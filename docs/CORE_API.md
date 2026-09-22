# 核心接口说明

## 组合与冻结评价

公开入口是 `crosslatent.ActivationDataset`、`fit`、`LinearReadout`、`evaluate` 和 `refit_leave_one_out`。模型 JSON 使用与项目名称无关的 `latent-readout/v1` schema，保存原始字典编号、系数、截距、阈值、字典标识，以及验证集选参轨迹。

验证指标：二分类为每条序列的正／负类别 F1 均值，再对序列等权平均；连续任务为所有 token 的 pooled R²。R² 标签方差为零时沿用历史引擎返回 0 的约定。二分类验证阈值为 0–1、步长 0.05。`lower` 与 `closest-half` 仅控制同一个 K 内的阈值并列；跨 K 同分优先较小 K。

稀疏预算限定为 1 到字典宽度。退化字典可能使历史引擎重复选到同一列，公开模型会保留唯一列，因此实际项数可能低于请求上限。该处理不改变已有非退化论文结果的选择规则，也不宣称固定 K 的全局子集最优。

`sequence_ids` 的跨划分检查用于发现明显文件混用，不能检查基础模型预训练污染、序列同源性或不同 ID 下相同序列。

## 共享 latent 与位置响应

```python
from crosslatent import LinearReadout
from crosslatent.interpretation import shared_latents, position_profile

left = LinearReadout.load("family_a.json")
right = LinearReadout.load("family_b.json")
shared = shared_latents([left, right])
# offsets 为每 token 的已注释位置，目标外为 -1。
profile = position_profile(activations, offsets, shared, n_positions=8)
```

仅允许比较同一字典标识、同一宽度的模型。profile 是按位置求平均的描述性激活，不自动建立生物标签、不推断基序、不提供显著性结论。

## 移除后重新拟合

`refit_leave_one_out` 固定传入模型的 latent 集合，不补选。完整模型与每个删减模型均在训练激活上重新拟合所有剩余系数和截距；二分类各自按 closest-half 规则验证选阈值，随后在同一测试集评价。支持仅剩截距的模型。输出点估计，不自动增加置信区间。

这是有真实训练激活时的通用入口，不自动恢复历史缓存中的训练充分统计量。

## 模型读出干预

```python
import torch
from crosslatent.interpretation import readout_effects

head.eval()
effects = readout_effects(
    hidden,                 # [N, D]，原始末层表示
    features,               # [N, M]，同一表示的固定 SAE 激活
    decoder,                # [D, M]，每列对应一个 latent
    shared_ids,
    head,                   # 已加载的外部模型输出头
    token_ids.long(),        # [N]，真实氨基酸的词表编号
)
```

直接计算 `hidden - features[:, ids] @ decoder[:, ids].T`，保留原始 SAE 重建残差，不删除 decoder bias。传入张量必须采用同一原始表示坐标；不支持将归一化空间的 decoder 直接用于未经对应缩放的原始表示。函数不按核心位置遮罩干预；返回逐 token 的 log probability、margin、KL 和 top-1 变化，区域汇总由调用者的已有注释决定。该接口检验模型读出作用，不等于生化功能干预。

## 原始流式引擎

`crosslatent._core.neuron_boosting_seq` 保留历史 `build_streaming_context`、`StreamingFeatureEvalParams` 和 `evaluate_feature_streaming` 等入口。调用者提供满足 `extract_embeddings(sequences, layer, batch_size)` 的已加载 embedder，以及具有 `encode(tensor)` 的 SAE。不会根据本机目录寻找或安装外部项目。

流式实现多次遍历数据，调用者必须保证每次的序列顺序、编码结果及批次组成一致。其原始坐标恢复拟合 y_boost；根据 `closed_form_full_pass_token_threshold` 决定全量或单批恢复。使用者应显式指定此阈值，不能把单批恢复描述为全训练集标签 OLS。统一 NPZ CLI 当前使用内存引擎，没有将历史流式行为悄悄替换成新算法。

