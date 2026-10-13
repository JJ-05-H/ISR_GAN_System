# evaluate.py
"""
ISR-GAN 量化评估脚本：在 test / val 划分上计算 PSNR / SSIM，并做三方对比。

对比对象：
    bicubic   双三次插值上采样（下界基线，无需权重）
    pretrain  MSE 预训练生成器（checkpoints/generator_pretrain.pth）
    gan       GAN 对抗微调生成器（默认 checkpoints/generator_gan_epoch20.pth）

为什么需要这个脚本：
    项目 README 原先给出了 PSNR/SSIM 数字，但仓库里没有任何指标计算代码，
    无法复现也无法证明。本脚本把「指标从哪来」变成可执行、可复现的流程。

指标口径（与超分领域惯例一致）：
    - 基础 PSNR/SSIM 在 RGB 上计算（对应图像最终观感）
    - 同时给出 YCbCr 的 Y 通道（亮度）结果，这是 SR 论文常用的报告口径
    - 可选 shave 边界（默认 4px）：生成图像的边缘若干像素不可靠，标准做法是裁掉后再算
    - PSNR 的 MAX 取 255，按 uint8 输出计算，与推理链路一致

用法：
    python evaluate.py                                  # 评估 test，跑三方对比
    python evaluate.py --split val                      # 换划分
    python evaluate.py --weights bicubic gan            # 只跑指定对象
    python evaluate.py --data-root processed --limit 100
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config                      # noqa: E402
from src.models.generator import Generator     # noqa: E402

# 各评估对象的默认权重路径
WEIGHT_PATHS = {
    "pretrain": "checkpoints/generator_pretrain.pth",
    "gan": "checkpoints/generator_gan_epoch20.pth",
    "gan_final": "checkpoints/generator_final.pth",
}


# ---------------- LPIPS（感知指标） ----------------

_LPIPS_FN = None


def get_lpips():
    """懒加载 LPIPS（AlexNet 骨干）。

    为什么必须测感知指标：
        PSNR/SSIM 是保真度指标，而对抗训练的卖点正是「牺牲保真度换感知质量」。
        只用 PSNR 评价 GAN 会系统性地低估它——这是超分领域的经典权衡。
        LPIPS 越低越接近人眼感知，是判断 GAN 是否真正起作用的关键补充。
    首次调用会下载 AlexNet 预训练权重（约 233MB）并缓存。
    """
    global _LPIPS_FN
    if _LPIPS_FN is None:
        try:
            import lpips
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "未安装 lpips，无法计算感知指标。请执行：\n"
                "  pip install --no-deps lpips\n"
                "（--no-deps 避免 pip 重装 torch）"
            ) from e
        _LPIPS_FN = lpips.LPIPS(net="alex", verbose=False)
        _LPIPS_FN.eval()
    return _LPIPS_FN


def to_lpips_tensor(img_rgb_uint8: np.ndarray):
    """uint8 RGB → LPIPS 需要的 [-1,1] NCHW tensor"""
    x = img_rgb_uint8.astype(np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def calc_lpips(sr_rgb_uint8: np.ndarray, hr_rgb_uint8: np.ndarray, fn) -> float:
    """LPIPS 距离（越低越接近人眼感知上的相似）。

    注意 detach()：lpips 的返回值带梯度，不 detach 会在累加时告警并占用显存。
    """
    with torch.no_grad():
        d = fn(to_lpips_tensor(sr_rgb_uint8), to_lpips_tensor(hr_rgb_uint8))
    return float(d.detach().mean())


# ---------------- 指标实现 ----------------

def to_uint8(img_float01: np.ndarray) -> np.ndarray:
    """float [0,1] → uint8 [0,255]，四舍五入"""
    return np.clip(np.rint(img_float01 * 255.0), 0, 255).astype(np.uint8)


def calc_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """均方误差（uint8 输入，与 PSNR 的 MAX=255 口径配套）"""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def mse_to_psnr(mse: float) -> float:
    """MSE → PSNR(dB)；MSE=0 表示完全一致，返回 inf"""
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def calc_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """单张图的 PSNR(dB)

    注意：**不要对多张图的 PSNR 取算术平均**。
    纯色图块（全白/全黑）会得到 MSE=0 → PSNR=inf，
    直接平均 inf 会污染整体结果。
    数据集级别的汇总请用 aggregate_psnr()。
    """
    return mse_to_psnr(calc_mse(img1, img2))


def aggregate_psnr(mse_list) -> float:
    """数据集级别的 PSNR：先对全部样本求平均 MSE，再换算成 PSNR

    这是「按像素池化」的口径，对 MSE=0 的样本天然免疫
    （平均 MSE 只在所有样本都完全一致时才为 0）。
    但它对高误差样本赋予更高权重，数值会明显低于单张 PSNR 的算术平均。
    """
    if len(mse_list) == 0:
        return float("nan")
    return mse_to_psnr(float(np.mean(mse_list)))


def mean_psnr(mse_list) -> float:
    """数据集级别的 PSNR：先算每对图像的 PSNR，再取算术平均

    这是超分论文（Set5/Set14/DIV2K 等基准）最常用的报告口径，
    与 README 中「实验结果」表的数字口径一致。
    注意 MSE=0（纯色图块完全一致）会产生 inf，这里按惯例剔除后再平均，
    同时用 perfect_pairs 记录这类样本的数量。
    """
    if len(mse_list) == 0:
        return float("nan")
    finite = [mse_to_psnr(m) for m in mse_list if m > 0]
    if not finite:
        return float("inf")
    return float(np.mean(finite))


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def calc_ssim(img1: np.ndarray, img2: np.ndarray,
              kernel: np.ndarray | None = None) -> float:
    """SSIM（Wang et al. 2004，11x11 高斯窗 sigma=1.5）

    实现在单通道上进行；多通道时由调用方取平均。
    不依赖 scikit-image，避免新增重依赖。

    边界口径：使用 BORDER_REFLECT_101（镜像不含边缘像素），
    与 skimage.metrics.structural_similarity 的默认行为一致，
    便于用 scikit-image 复核本脚本的结果。
    注意 OpenCV 的 BORDER_REFLECT 是「含边缘像素」的对称反射，
    与 skimage 默认不同，这里刻意不用它。
    """
    if kernel is None:
        kernel = _gaussian_kernel()

    x = img1.astype(np.float64)
    y = img2.astype(np.float64)

    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2

    # 用 filter2D 做高斯加权均值，等价于论文里的滑窗卷积
    mu_x = cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    mu_y = cv2.filter2D(y, -1, kernel, borderType=cv2.BORDER_REFLECT_101)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = cv2.filter2D(x * x, -1, kernel,
                            borderType=cv2.BORDER_REFLECT_101) - mu_x2
    sigma_y2 = cv2.filter2D(y * y, -1, kernel,
                            borderType=cv2.BORDER_REFLECT_101) - mu_y2
    sigma_xy = cv2.filter2D(x * y, -1, kernel,
                            borderType=cv2.BORDER_REFLECT_101) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return float(ssim_map.mean())


def calc_ssim_multi(img1: np.ndarray, img2: np.ndarray,
                    kernel: np.ndarray | None = None) -> float:
    """多通道 SSIM：按通道平均（超分常用做法）"""
    if img1.ndim == 2:
        return calc_ssim(img1, img2, kernel)
    if kernel is None:
        kernel = _gaussian_kernel()
    return float(np.mean([calc_ssim(img1[:, :, c], img2[:, :, c], kernel)
                          for c in range(img1.shape[2])]))


def shave(img: np.ndarray, border: int) -> np.ndarray:
    """裁掉四周 border 像素（标准 SR 评测会裁掉边缘不可靠区域）"""
    if border <= 0:
        return img
    return img[border:-border, border:-border]


def to_y_channel(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """RGB(uint8) → YCbCr 的 Y 通道（BT.601，与 cv2.COLOR_RGB2YCrCb 一致）"""
    ycrcb = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2YCrCb)
    return ycrcb[:, :, 0]


# ---------------- 模型 / 推理 ----------------

def build_generator(weights_path: str, device):
    """按后端同样的方式加载生成器"""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到权重文件: {weights_path}")
    cfg = Config()
    gen = Generator(cfg).to(device)
    gen.load_state_dict(torch.load(weights_path, map_location=device))
    gen.eval()
    return gen


def load_pair(hr_path: str, lr_path: str):
    """读取 HR/LR 图像对，返回 RGB uint8"""
    hr = cv2.imread(hr_path)
    lr = cv2.imread(lr_path)
    if hr is None or lr is None:
        return None, None
    return (cv2.cvtColor(hr, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(lr, cv2.COLOR_BGR2RGB))


def lr_to_tensor(lr_rgb_uint8: np.ndarray, device):
    """与 backend/utils.py 一致的预处理：/255 → [-1,1] → NCHW"""
    x = lr_rgb_uint8.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def tensor_to_rgb_uint8(out_tensor) -> np.ndarray:
    """与 backend/utils.py 一致的后处理：(x*0.5+0.5)*255 → uint8

    注意：模型输出经过 tanh 落在 [-1,1]，这里裁剪后转 uint8，
    这一步的量化误差已实测为 0（见 README「评估口径」）。
    """
    img = out_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return to_uint8(img * 0.5 + 0.5)


# ---------------- 主流程 ----------------

def evaluate(data_root: str, split: str, targets: list[str], limit: int | None,
             shave_border: int, device, out_dir: str, save_images: int = 0,
             use_lpips: bool = True):
    index_path = Path(data_root) / f"{split}_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"找不到索引 {index_path}。请先运行预处理：\n"
            f"  python -m src.data.preprocess --source-dir <DIV2K目录> --splits {split}"
        )

    pairs = json.loads(index_path.read_text(encoding="utf-8"))["pairs"]
    if limit:
        pairs = pairs[:limit]
    print(f"[{split}] 待评估图像对: {len(pairs)}")

    lpips_fn = None
    if use_lpips:
        print("  加载 LPIPS (alex)...")
        lpips_fn = get_lpips()
        print("  已加载 LPIPS")

    # 载入模型（只加载实际要用的）
    models = {}
    for t in targets:
        if t == "bicubic":
            continue
        wpath = WEIGHT_PATHS.get(t, t)
        models[t] = build_generator(wpath, device)
        print(f"  已加载 {t}: {wpath}")

    kernel = _gaussian_kernel()
    # 存 MSE 而不是 PSNR：纯色图块会得到 MSE=0，若存 PSNR 就是 inf，
    # 对 inf 取平均会污染结果。最后用 aggregate_psnr 从平均 MSE 换算。
    accum = {t: {"mse_rgb": [], "ssim_rgb": [], "mse_y": [], "ssim_y": [],
                 "lpips": []}
             for t in targets}

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    saved = 0

    with torch.no_grad():
        for i, pair in enumerate(pairs, 1):
            hr, lr = load_pair(pair["hr_path"], pair["lr_path"])
            if hr is None:
                print(f"  [跳过] 无法读取: {pair['id']}")
                continue

            hr_h, hr_w = hr.shape[:2]

            for t in targets:
                if t == "bicubic":
                    sr = cv2.resize(lr, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
                else:
                    x = lr_to_tensor(lr, device)
                    out = models[t](x)
                    sr = tensor_to_rgb_uint8(out)

                    # 尺寸校验：模型输出应为输入的 4 倍
                    if sr.shape[:2] != (hr_h, hr_w):
                        sr = cv2.resize(sr, (hr_w, hr_h),
                                        interpolation=cv2.INTER_CUBIC)

                hr_eval = shave(hr, shave_border)
                sr_eval = shave(sr, shave_border)

                accum[t]["mse_rgb"].append(calc_mse(sr_eval, hr_eval))
                accum[t]["ssim_rgb"].append(calc_ssim_multi(sr_eval, hr_eval, kernel))
                accum[t]["mse_y"].append(
                    calc_mse(to_y_channel(sr_eval), to_y_channel(hr_eval)))
                accum[t]["ssim_y"].append(
                    calc_ssim(to_y_channel(sr_eval), to_y_channel(hr_eval), kernel))
                if lpips_fn is not None:
                    accum[t]["lpips"].append(calc_lpips(sr_eval, hr_eval, lpips_fn))

                # 可选：保存前若干张对比图
                if save_images and i <= save_images and t == targets[-1]:
                    canvas = np.hstack([hr, cv2.resize(lr, (hr_w, hr_h),
                                                       interpolation=cv2.INTER_NEAREST), sr])
                    cv2.imwrite(str(out_dir_path / f"{pair['id']}_{t}_hr_lr_sr.png"),
                                cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                    saved += 1

            if i % 50 == 0 or i == len(pairs):
                msg = " | ".join(
                    f"{t}: PSNR {aggregate_psnr(accum[t]['mse_rgb']):.2f}"
                    for t in targets)
                print(f"  [{i}/{len(pairs)}] {msg}")

    # 汇总
    results = {}
    for t in targets:
        a = accum[t]
        if not a["mse_rgb"]:
            continue
        results[t] = {
            "n": len(a["mse_rgb"]),
            # 两种口径都给：mean_psnr 对应 SR 论文惯例与 README 原表，
            # pooled_psnr 对应按像素池化（对高误差样本更敏感）
            "psnr_rgb": mean_psnr(a["mse_rgb"]),
            "psnr_rgb_pooled": aggregate_psnr(a["mse_rgb"]),
            "ssim_rgb": float(np.mean(a["ssim_rgb"])),
            "psnr_y": mean_psnr(a["mse_y"]),
            "psnr_y_pooled": aggregate_psnr(a["mse_y"]),
            "ssim_y": float(np.mean(a["ssim_y"])),
            "psnr_rgb_std": float(np.std([
                p for p in (mse_to_psnr(m) for m in a["mse_rgb"])
                if np.isfinite(p)] or [0.0])),
            "perfect_pairs": int(sum(1 for m in a["mse_rgb"] if m == 0)),
            # 感知指标：越低越好（与 PSNR/SSIM 方向相反）
            "lpips": float(np.mean(a["lpips"])) if a["lpips"] else None,
        }

    return results


def print_table(results: dict, split: str, shave_border: int):
    has_lpips = any(r.get("lpips") is not None for r in results.values())
    print()
    print("=" * 86)
    print(f"评估结果  split={split}  shave={shave_border}px")
    print("PSNR/SSIM 越高越好；LPIPS 越低越好（感知指标，与 PSNR 方向相反）")
    print("口径 A「单张均值」：先算每对图的 PSNR 再平均 —— SR 论文惯例")
    print("口径 B「池化」    ：先对全测试集求平均 MSE 再换算 —— 对高误差样本更敏感")
    print("=" * 86)
    header = (f"{'方法':<11}{'PSNR(RGB)A':>11}{'PSNR(RGB)B':>11}"
              f"{'SSIM(RGB)':>11}{'PSNR(Y)A':>10}{'SSIM(Y)':>9}"
              + (f"{'LPIPS':>9}" if has_lpips else "") + f"{'n':>7}")
    print(header)
    print("-" * 86)
    order = ["bicubic", "pretrain", "gan", "gan_final"]
    for t in [x for x in order if x in results] + \
             [x for x in results if x not in order]:
        r = results[t]
        row = (f"{t:<11}{r['psnr_rgb']:>11.3f}"
               f"{r.get('psnr_rgb_pooled', float('nan')):>11.3f}"
               f"{r['ssim_rgb']:>11.4f}{r['psnr_y']:>10.3f}{r['ssim_y']:>9.4f}")
        if has_lpips:
            lp = r.get("lpips")
            row += f"{lp:>9.4f}" if lp is not None else f"{'-':>9}"
        print(row + f"{r['n']:>7}")
    print("-" * 86)

    # 相对双三次基线的增益
    if "bicubic" in results:
        base = results["bicubic"]
        print("相对双三次基线的增益：")
        base_lp = base.get("lpips")
        for t in results:
            if t == "bicubic":
                continue
            dp = results[t]["psnr_rgb"] - base["psnr_rgb"]
            ds = results[t]["ssim_rgb"] - base["ssim_rgb"]
            line = f"  {t:<10} PSNR {dp:+.3f} dB   SSIM {ds:+.4f}"
            lp = results[t].get("lpips")
            if base_lp is not None and lp is not None:
                line += f"   LPIPS {lp - base_lp:+.4f}（负值为更好）"
            print(line)
    perfect = {t: results[t].get("perfect_pairs", 0) for t in results}
    if any(perfect.values()):
        print(f"注：完全一致的图块数（MSE=0，已从口径 A 中剔除）: {perfect}")
    if has_lpips:
        print("解读提示：若某方法 PSNR 偏低但 LPIPS 也更低，说明它走的是")
        print("          「牺牲保真度换感知质量」路线——这正是对抗训练的预期收益。")
    print("=" * 86)


def save_results(results: dict, split: str, out_dir: str, shave_border: int) -> Path:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "split": split,
        "shave_border": shave_border,
        "timestamp": ts,
        "results": results,
    }
    json_path = out_dir_path / f"eval_{split}_{ts}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    csv_path = out_dir_path / f"eval_{split}_{ts}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["method", "psnr_rgb_mean", "psnr_rgb_pooled", "ssim_rgb",
                    "psnr_y_mean", "psnr_y_pooled", "ssim_y", "lpips",
                    "psnr_rgb_std", "perfect_pairs", "n"])
        for t, r in results.items():
            lp = r.get("lpips")
            w.writerow([t, f"{r['psnr_rgb']:.4f}", f"{r.get('psnr_rgb_pooled', float('nan')):.4f}",
                        f"{r['ssim_rgb']:.6f}",
                        f"{r['psnr_y']:.4f}", f"{r.get('psnr_y_pooled', float('nan')):.4f}",
                        f"{r['ssim_y']:.6f}",
                        f"{lp:.6f}" if lp is not None else "",
                        f"{r['psnr_rgb_std']:.4f}", r.get("perfect_pairs", 0),
                        r["n"]])

    print(f"\n结果已保存：\n  {json_path}\n  {csv_path}")
    return json_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="ISR-GAN 量化评估（PSNR / SSIM / LPIPS）")
    p.add_argument("--data-root", default="processed",
                   help="processed 目录（含 *_index.json），默认 processed")
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="评估哪个划分，默认 test")
    p.add_argument("--weights", nargs="+", default=["bicubic", "pretrain", "gan"],
                   help="评估对象：bicubic / pretrain / gan / gan_final，"
                        "也可以直接给 .pth 路径")
    p.add_argument("--limit", type=int, default=None,
                   help="只评估前 N 对，便于快速验证")
    p.add_argument("--shave", type=int, default=4,
                   help="四周裁掉的边界像素数，默认 4（标准 SR 评测做法）")
    p.add_argument("--out-dir", default="results", help="结果输出目录")
    p.add_argument("--save-images", type=int, default=0,
                   help="保存前 N 张 HR/LR/SR 对比图")
    p.add_argument("--no-lpips", action="store_true",
                   help="跳过 LPIPS 感知指标（默认计算；它只做一次 AlexNet 前向，"
                        "约 25ms/对，是判断 GAN 是否真正起作用的关键指标）")
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cpu") if args.cpu else Config().device
    print(f"设备: {device}")

    results = evaluate(
        data_root=args.data_root,
        split=args.split,
        targets=args.weights,
        limit=args.limit,
        shave_border=args.shave,
        device=device,
        out_dir=args.out_dir,
        save_images=args.save_images,
        use_lpips=not args.no_lpips,
    )
    if not results:
        print("没有可用的评估结果。")
        return 1

    print_table(results, args.split, args.shave)
    save_results(results, args.split, args.out_dir, args.shave)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
