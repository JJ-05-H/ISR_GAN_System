# tests/test_model.py
"""生成器结构与推理链路的测试

不依赖数据集；权重文件缺失时相关用例自动跳过
（checkpoints/*.pth 被 .gitignore 白名单放行，CI 上通常存在，
 但一旦缺失也不应让整个测试套件失败）。
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config  # noqa: E402
from src.models.discriminator import ConditionalDiscriminator  # noqa: E402
from src.models.generator import Generator  # noqa: E402
from evaluate import lr_to_tensor, tensor_to_rgb_uint8  # noqa: E402


# ---------------- 生成器结构 ----------------

@pytest.fixture(scope="module")
def generator():
    torch.manual_seed(0)
    g = Generator(Config())
    g.eval()
    return g


def test_generator_upscales_by_4(generator):
    """4 倍超分：输出尺寸必须是输入的 4 倍"""
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = generator(x)
    assert y.shape == (1, 3, 256, 256)


def test_generator_handles_various_input_sizes(generator):
    """PixelShuffle 要求尺寸可被 4 整除；验证常见尺寸可用"""
    for hw in [(32, 32), (64, 48), (128, 96)]:
        x = torch.randn(1, 3, *hw)
        with torch.no_grad():
            y = generator(x)
        assert y.shape == (1, 3, hw[0] * 4, hw[1] * 4)


def test_generator_output_range_is_tanh(generator):
    """输出经 tanh，必须落在 [-1, 1]"""
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = generator(x)
    assert y.min() >= -1.0 - 1e-6
    assert y.max() <= 1.0 + 1e-6


def test_generator_param_count_matches_checkpoint(generator):
    """参数量应与仓库中权重文件自洽（约 82 万，FP32 约 3.16MB）"""
    n = sum(p.numel() for p in generator.parameters())
    assert 700_000 < n < 900_000, f"参数量异常: {n}"


# ---------------- 判别器结构 ----------------

def test_discriminator_accepts_4_channels():
    """条件判别器输入 4 通道（RGB + 掩码），输出 1 维 logit"""
    d = ConditionalDiscriminator()
    d.eval()
    img = torch.randn(2, 3, 256, 256)
    mask = torch.ones(2, 1, 256, 256)
    with torch.no_grad():
        out = d(img, mask)
    assert out.shape == (2, 1)


def test_discriminator_outputs_logits_not_probabilities():
    """判别器返回 logits（训练用 BCEWithLogitsLoss），
    所以取值不应被限制在 [0,1]；这是区分它与「概率」的关键特征。
    """
    torch.manual_seed(1)
    d = ConditionalDiscriminator()
    d.eval()
    img = torch.randn(8, 3, 64, 64) * 3
    mask = torch.ones(8, 1, 64, 64)
    with torch.no_grad():
        out = d(img, mask)
    # logits 通常会出现负值或大于 1 的值
    assert out.min() < 0 or out.max() > 1, "输出像是概率而非 logits"


# ---------------- 推理链路（与 backend/utils.py 口径一致） ----------------

def test_lr_to_tensor_normalization():
    """LR uint8 -> [-1,1] tensor，且通道顺序被保留"""
    lr = np.zeros((8, 8, 3), dtype=np.uint8)
    lr[:, :, 0] = 255          # R 通道全白
    t = lr_to_tensor(lr, torch.device("cpu"))
    assert t.shape == (1, 3, 8, 8)
    assert torch.allclose(t[0, 0], torch.ones(8, 8))          # R -> 1
    assert torch.allclose(t[0, 1], -torch.ones(8, 8))         # G -> -1


def test_tensor_to_rgb_uint8_roundtrip():
    """[-1,1] tensor -> uint8 的还原口径"""
    t = torch.zeros(1, 3, 4, 4)
    out = tensor_to_rgb_uint8(t)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.uint8
    # 0 对应 (0*0.5+0.5)*255 = 127.5 -> 四舍五入 128
    assert (out == 128).all()


def test_inference_pipeline_end_to_end(generator):
    """完整链路：uint8 LR -> tensor -> 模型 -> uint8 SR，尺寸放大 4 倍"""
    rng = np.random.default_rng(0)
    lr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    t = lr_to_tensor(lr, torch.device("cpu"))
    with torch.no_grad():
        out = generator(t)
    sr = tensor_to_rgb_uint8(out)
    assert sr.shape == (256, 256, 3)
    assert sr.dtype == np.uint8


# ---------------- 权重加载 ----------------

@pytest.mark.parametrize("weights", [
    "checkpoints/generator_pretrain.pth",
    "checkpoints/generator_gan_epoch20.pth",
])
def test_checkpoint_loads_into_generator(weights):
    """仓库中的权重必须能严格加载进当前 Generator 结构"""
    path = os.path.join(ROOT, weights)
    if not os.path.exists(path):
        pytest.skip(f"权重不存在（被 gitignore 或未训练）: {weights}")

    g = Generator(Config())
    state = torch.load(path, map_location="cpu")
    # strict=True：键名或形状不匹配会直接抛错
    g.load_state_dict(state)
    assert isinstance(g, Generator)


def test_optional_skimage_crosscheck():
    """如果环境里有 scikit-image，用它交叉验证我们的 SSIM 实现

    没有该依赖时跳过——它只是复核手段，不是运行依赖。
    """
    skimage = pytest.importorskip("skimage.metrics")
    from evaluate import calc_ssim

    rng = np.random.default_rng(7)
    a = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    b = np.clip(a.astype(np.int32) + rng.integers(-10, 11, a.shape), 0, 255).astype(np.uint8)

    mine = calc_ssim(a, b)
    theirs = skimage.structural_similarity(a, b, data_range=255, gaussian_weights=True,
                                          sigma=1.5, use_sample_covariance=False)
    assert mine == pytest.approx(theirs, abs=2e-3), \
        f"与 scikit-image 不一致: mine={mine}, skimage={theirs}"
