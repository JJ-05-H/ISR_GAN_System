# tests/test_metrics.py
"""评估指标实现的正确性测试

SSIM 的向量化实现（cv2.filter2D）必须与朴素滑窗定义逐点等价，
否则算出来的分数没有意义。这里用暴力实现做交叉验证。
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluate import (  # noqa: E402
    _gaussian_kernel,
    aggregate_psnr,
    calc_mse,
    calc_psnr,
    calc_ssim,
    calc_ssim_multi,
    mean_psnr,
    mse_to_psnr,
    shave,
    to_uint8,
    to_y_channel,
)
import cv2  # noqa: E402


def ssim_bruteforce(img1, img2, kernel):
    """按 Wang et al. 2004 的定义朴素逐窗计算（慢，仅用于验证）

    边界用 np.pad(mode="reflect")（镜像不含边缘像素），
    与实现里的 cv2.BORDER_REFLECT_101 以及 skimage 默认口径一致。
    """
    k = kernel.shape[0]
    pad = k // 2
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)

    xp = np.pad(x, pad, mode="reflect")
    yp = np.pad(y, pad, mode="reflect")

    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2

    h, w = x.shape
    total = 0.0
    for i in range(h):
        for j in range(w):
            wx = xp[i:i + k, j:j + k]
            wy = yp[i:i + k, j:j + k]
            mu_x = (wx * kernel).sum()
            mu_y = (wy * kernel).sum()
            mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
            sig_x2 = (wx * wx * kernel).sum() - mu_x2
            sig_y2 = (wy * wy * kernel).sum() - mu_y2
            sig_xy = (wx * wy * kernel).sum() - mu_xy
            total += (((2 * mu_xy + C1) * (2 * sig_xy + C2)) /
                      ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)))
    return total / (h * w)


# ---------------- PSNR ----------------

def test_psnr_identical_is_infinite():
    a = np.full((16, 16, 3), 128, dtype=np.uint8)
    assert calc_psnr(a, a) == float("inf")


def test_psnr_known_value():
    """常数差 d 的 PSNR = 20*log10(255/d)"""
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.full((8, 8, 3), 10, dtype=np.uint8)
    expected = 20 * np.log10(255.0 / 10.0)
    assert calc_psnr(a, b) == pytest.approx(expected, rel=1e-9)


def test_psnr_symmetric():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    assert calc_psnr(a, b) == pytest.approx(calc_psnr(b, a))


# ---------------- 数据集级 PSNR（回归：inf 污染） ----------------

def test_calc_mse_matches_manual():
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.full((4, 4), 10, dtype=np.uint8)
    assert calc_mse(a, b) == pytest.approx(100.0)


def test_mse_to_psnr_zero_is_infinite():
    assert mse_to_psnr(0.0) == float("inf")


def test_aggregate_psnr_equals_mean_mse_conversion():
    """数据集级 PSNR 必须等价于「平均 MSE 再换算」"""
    mses = [1.0, 4.0, 16.0]
    expected = mse_to_psnr(np.mean(mses))
    assert aggregate_psnr(mses) == pytest.approx(expected)


def test_aggregate_psnr_immune_to_perfect_pairs():
    """回归测试：纯色图块会得到 MSE=0（单张 PSNR=inf），
    若对 PSNR 取平均会被 inf 污染；aggregate_psnr 必须不受影响。
    """
    mses = [0.0, 0.0, 100.0]          # 两张完全一致 + 一张有误差
    got = aggregate_psnr(mses)

    # 正确结果 = 由平均 MSE(33.33) 换算
    assert got == pytest.approx(mse_to_psnr(100.0 / 3.0))
    assert np.isfinite(got)

    # 对照：错误做法（对单张 PSNR 取平均）会得到 inf
    naive = float(np.mean([mse_to_psnr(m) for m in mses]))
    assert naive == float("inf")
    assert got != naive


def test_aggregate_psnr_all_perfect_is_infinite():
    assert aggregate_psnr([0.0, 0.0]) == float("inf")


def test_aggregate_psnr_empty_is_nan():
    assert np.isnan(aggregate_psnr([]))


# ---------------- 两种聚合口径（回归：容易混淆） ----------------

def test_mean_psnr_equals_arithmetic_mean_of_per_image():
    """口径 A：先算每张图的 PSNR，再取算术平均"""
    mses = [1.0, 4.0, 16.0]
    expected = np.mean([mse_to_psnr(m) for m in mses])
    assert mean_psnr(mses) == pytest.approx(expected)


def test_two_conventions_differ_and_pooled_is_lower():
    """口径 A（单张均值）与口径 B（池化）必须不同，
    且池化口径因为按像素加权、受高误差样本影响更大，数值更低。
    这正是本项目的实测现象，必须有测试固定住，避免报告时把两个数混用。
    """
    mses = [1.0, 100.0, 10000.0]
    a = mean_psnr(mses)
    b = aggregate_psnr(mses)
    assert a > b
    assert a == pytest.approx(np.mean([mse_to_psnr(m) for m in mses]))
    assert b == pytest.approx(mse_to_psnr(np.mean(mses)))


def test_mean_psnr_excludes_perfect_pairs():
    """MSE=0 会产生 inf，口径 A 按惯例剔除后再平均"""
    mses = [0.0, 1.0, 4.0]
    expected = np.mean([mse_to_psnr(1.0), mse_to_psnr(4.0)])
    assert mean_psnr(mses) == pytest.approx(expected)
    assert np.isfinite(mean_psnr(mses))


def test_mean_psnr_all_perfect_is_infinite():
    assert mean_psnr([0.0, 0.0]) == float("inf")


def test_mean_psnr_empty_is_nan():
    assert np.isnan(mean_psnr([]))


def test_pooled_psnr_uses_all_samples_including_perfect():
    """口径 B 把 MSE=0 也计入平均 MSE（这正是它不怕 inf 的原因）"""
    mses = [0.0, 100.0]
    assert aggregate_psnr(mses) == pytest.approx(mse_to_psnr(50.0))


def test_real_images_produce_finite_aggregate():
    """真实场景：一组有误差的图，聚合结果必须有限且合理"""
    rng = np.random.default_rng(11)
    mses = []
    for _ in range(10):
        a = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        b = np.clip(a.astype(np.int32) + rng.integers(-8, 9, a.shape),
                    0, 255).astype(np.uint8)
        mses.append(calc_mse(a, b))
    v = aggregate_psnr(mses)
    assert np.isfinite(v)
    assert 20.0 < v < 50.0


# ---------------- SSIM ----------------

def test_ssim_identical_is_one():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    assert calc_ssim(a, a) == pytest.approx(1.0, abs=1e-6)


def test_ssim_matches_bruteforce_small():
    """小图上与朴素滑窗实现逐点一致"""
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, (24, 24), dtype=np.uint8)
    b = rng.integers(0, 256, (24, 24), dtype=np.uint8)
    kernel = _gaussian_kernel()
    fast = calc_ssim(a, b, kernel)
    slow = ssim_bruteforce(a, b, kernel)
    assert fast == pytest.approx(slow, rel=1e-6)


def test_ssim_matches_bruteforce_smooth():
    """平滑图像（更接近真实 HR/LR）也要一致"""
    x = np.linspace(0, 255, 40)
    a = np.tile(x, (30, 1)).astype(np.uint8)
    b = np.clip(a.astype(np.int32) + 7, 0, 255).astype(np.uint8)
    kernel = _gaussian_kernel()
    assert calc_ssim(a, b, kernel) == pytest.approx(
        ssim_bruteforce(a, b, kernel), rel=1e-6)


def test_ssim_range():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        b = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        v = calc_ssim(a, b)
        assert -1.0 <= v <= 1.0


def test_ssim_decreases_with_noise():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    noisy_small = np.clip(a.astype(np.int32) + rng.integers(-5, 6, a.shape),
                          0, 255).astype(np.uint8)
    noisy_big = np.clip(a.astype(np.int32) + rng.integers(-40, 41, a.shape),
                        0, 255).astype(np.uint8)
    assert calc_ssim(a, noisy_small) > calc_ssim(a, noisy_big)


def test_ssim_multi_averages_channels():
    rng = np.random.default_rng(4)
    a = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    kernel = _gaussian_kernel()
    per_ch = np.mean([calc_ssim(a[:, :, c], b[:, :, c], kernel) for c in range(3)])
    assert calc_ssim_multi(a, b, kernel) == pytest.approx(per_ch, rel=1e-9)


def test_ssim_multi_accepts_grayscale():
    rng = np.random.default_rng(5)
    a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    b = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    assert calc_ssim_multi(a, b) == pytest.approx(calc_ssim(a, b), rel=1e-9)


# ---------------- 辅助函数 ----------------

def test_to_uint8_rounds_and_clips():
    assert to_uint8(np.array([[[0.0, 1.0, 0.502]]]))[0, 0].tolist() == [0, 255, 128]
    assert to_uint8(np.array([[[-0.5, 1.5]]]))[0, 0].tolist() == [0, 255]


def test_shave_removes_border():
    a = np.arange(36).reshape(6, 6)
    out = shave(a, 1)
    assert out.shape == (4, 4)
    assert out[0, 0] == a[1, 1]


def test_shave_zero_is_noop():
    a = np.arange(16).reshape(4, 4)
    assert np.array_equal(shave(a, 0), a)


def test_to_y_channel_shape_and_range():
    rng = np.random.default_rng(6)
    rgb = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    y = to_y_channel(rgb)
    assert y.shape == (16, 16)
    assert y.min() >= 0 and y.max() <= 255


def test_to_y_channel_matches_cv2():
    """Y 通道口径必须与 OpenCV 的 RGB2YCrCb 一致"""
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    y = to_y_channel(rgb)
    expected = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    assert np.array_equal(y, expected)
