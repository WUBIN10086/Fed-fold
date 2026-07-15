"""
FedAvg 聚合：把多个 client 各自微调得到的 SoloSeq checkpoint 权重做（加权）平均，
输出一个全局权重 .pt。

产出的 .pt 是 AlphaFold 层级的纯参数字典（与官方 seq_model_esm1b_ptm.pt 同构），
因此既可用于下一轮训练初始化，也可直接用于推理：
  - 训练初始化：train_openfold.py ... --resume_from_ckpt <global.pt> --resume_model_weights_only True
  - 推理：       run_pretrained_openfold.py ... --openfold_checkpoint_path <global.pt>

权重提取逻辑与 run_pretrained_openfold.py 保持一致：优先取 EMA 参数
（d["ema"]["params"]），这正是推理时实际加载的那套权重。

用法：
  # 等权平均 5 个 client（checkpoint 可以是 DeepSpeed 的 .ckpt 目录，也可以是 .pt 文件）
  python scripts/fedavg_aggregate.py \
      --checkpoints \
          data/.../client_0/output_dir/checkpoints/epoch=0-step=10000.ckpt \
          data/.../client_1/output_dir/checkpoints/epoch=0-step=10000.ckpt \
          ... \
      --output data/.../fedavg_round1/global_model.pt

  # 按各 client 训练样本数加权（标准 FedAvg：w_k ∝ n_k）
  python scripts/fedavg_aggregate.py --checkpoints c0 c1 c2 c3 c4 \
      --weights 120 95 140 88 102 \
      --output global_model.pt
"""
import argparse
import os
import tempfile
from pathlib import Path

import torch
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)


def load_client_params(ckpt_path: str) -> dict:
    """从一个 client checkpoint 提取 AlphaFold 层级的参数字典（优先 EMA）。"""
    p = Path(ckpt_path)

    if p.is_dir():
        # DeepSpeed / Lightning 的 .ckpt 目录：先转成 fp32 再读
        with tempfile.TemporaryDirectory() as td:
            converted = os.path.join(td, "converted.pt")
            convert_zero_checkpoint_to_fp32_state_dict(str(p), converted)
            d = torch.load(converted, map_location="cpu")
    else:
        d = torch.load(str(p), map_location="cpu")

    # 1) 训练 checkpoint：EMA 参数就是推理用的那套
    if isinstance(d, dict) and "ema" in d and isinstance(d["ema"], dict) and "params" in d["ema"]:
        return d["ema"]["params"]
    # 2) Lightning state_dict：键带 'model.' 前缀，剥掉得到 AlphaFold 层级键
    if isinstance(d, dict) and "state_dict" in d:
        sd = d["state_dict"]
        return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    # 3) 已经是纯参数字典（如官方 base 权重）
    return d


def fedavg(param_dicts: list[dict], weights: list[float]) -> dict:
    """对多套参数做加权平均。浮点张量按权重平均，非浮点缓冲区取第一个 client 的值。"""
    ref_keys = set(param_dicts[0].keys())
    for i, pd in enumerate(param_dicts[1:], start=1):
        if set(pd.keys()) != ref_keys:
            only_ref = ref_keys - set(pd.keys())
            only_i = set(pd.keys()) - ref_keys
            raise ValueError(
                f"client #{i} 的参数键与 client #0 不一致；"
                f"仅在 #0: {list(only_ref)[:5]}...  仅在 #{i}: {list(only_i)[:5]}..."
            )

    wsum = float(sum(weights))
    out = {}
    for k in param_dicts[0].keys():
        t0 = param_dicts[0][k]
        if torch.is_floating_point(t0):
            acc = torch.zeros_like(t0, dtype=torch.float64)
            for w, pd in zip(weights, param_dicts):
                acc += float(w) * pd[k].to(torch.float64)
            out[k] = (acc / wsum).to(t0.dtype)
        else:
            # 整型 buffer（如各种 index）各 client 相同，直接沿用第一个
            out[k] = t0.clone()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="各 client 的 checkpoint（DeepSpeed .ckpt 目录 或 .pt 文件），空格分隔",
    )
    parser.add_argument(
        "--output", required=True,
        help="输出的全局权重 .pt 路径",
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="各 client 的聚合权重（数量需与 --checkpoints 一致）。默认等权。"
             "标准 FedAvg 用各 client 训练样本数。",
    )
    args = parser.parse_args()

    n = len(args.checkpoints)
    if args.weights is None:
        weights = [1.0] * n
    else:
        if len(args.weights) != n:
            raise ValueError(f"--weights 数量({len(args.weights)}) 必须等于 checkpoint 数量({n})")
        weights = args.weights

    print(f"聚合 {n} 个 client，权重={weights}")
    param_dicts = []
    for i, ckpt in enumerate(args.checkpoints):
        print(f"  [{i}] 读取 {ckpt}")
        param_dicts.append(load_client_params(ckpt))

    print("正在做加权平均...")
    global_params = fedavg(param_dicts, weights)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(global_params, str(out_path))
    print(f"完成，全局权重已保存到: {out_path}  (参数张量数: {len(global_params)})")


if __name__ == "__main__":
    main()
