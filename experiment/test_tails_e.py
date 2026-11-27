"""补尾巴 E 组：跨设备的比较（RE2、RE3）。

和 CPU 存下来的数据比的那些断言，在 GPU 上改成按验收的阈值比；同一台设备上算出来的两份
结果（CPU 也好、GPU 也好）照旧逐位比。比法由两份输入各自所在的设备决定，入口是
``step6_support.exact``。

这里锁的是它的判别力：同一台设备上改坏一个位，比较必须失败；跨设备时改坏一个远大于阈值的量，
比较必须失败，差在舍入量级则通过。
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opencfd_jax.src import config

config.enable_x64()

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "experiment"):
    sys.path.insert(0, str(path))

import step6_support as support  # noqa: E402

SHAPE = (5, 2, 2, 2)
GPU = (("gpu", 0),)
CPU = (("cpu", 0),)


def _field():
    return np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE) + 1.0


def test_numpy_data_counts_as_computed_on_the_cpu():
    assert support.computed_on(np.zeros(3)) == CPU


def test_exact_is_bitwise_on_the_same_device():
    """缺省设备上算的两份（这台机器上是 CPU 就是 CPU，是 GPU 就是 GPU）逐位比；
    翻转一个最低位就报错。取回主机以后再比，也还是逐位。"""
    base = _field()
    on_device = jnp.asarray(base) * 2.0
    again = jnp.asarray(base) * 2.0
    support.exact(on_device, again, "同一台设备")
    support.exact(np.asarray(on_device), base * 2.0, "取回主机以后")
    flipped = again.at[0, 0, 0, 0].set(jnp.nextafter(again[0, 0, 0, 0], jnp.inf))
    assert np.any(np.asarray(flipped) != np.asarray(on_device))
    with pytest.raises(AssertionError):
        support.exact(flipped, on_device, "翻转最低位")


def test_exact_uses_the_threshold_across_devices():
    """Covers AE12。一份是存下来的 CPU 数据，另一份在 GPU 上算的：差在舍入量级就通过，
    差 1e-6 就报错。设备用 ``got_on`` 指定，没有 GPU 的机器上也能跑。"""
    want = _field()
    support.exact(want * (1.0 + 1e-16), want, "舍入量级", got_on=GPU)
    with pytest.raises(AssertionError, match="超出容差"):
        support.exact(want * (1.0 + 1e-6), want, "改坏", got_on=GPU)
    support.exact(want, want * (1.0 + 1e-16), "算的在右边", want_on=GPU)


def test_exact_compares_a_flat_sample_as_one_component():
    """一维的抽样没有分量可分：跨设备时整条按一个分量算容差。"""
    want = np.arange(1.0, 13.0)
    support.exact(want * (1.0 + 1e-16), want, "一维", got_on=GPU)
    with pytest.raises(AssertionError, match="超出容差"):
        support.exact(want * (1.0 + 1e-6), want, "一维", got_on=GPU)


@pytest.mark.skipif(all(device.platform == "cpu" for device in jax.devices()),
                    reason="没有 GPU")
def test_exact_sees_where_a_real_array_lives():
    """有 GPU 的机器上：同一个数放在两张设备上，``exact`` 认得出哪一份在 GPU 上。"""
    ones = np.ones(SHAPE)
    on_cpu = jax.device_put(ones, jax.devices("cpu")[0])
    on_gpu = jax.device_put(ones, jax.devices()[0])
    assert support.computed_on(on_cpu) == CPU
    assert support.computed_on(on_gpu) != CPU
    support.exact(on_gpu, on_cpu, "两张设备上的同一个数")
    with pytest.raises(AssertionError, match="超出容差"):
        support.exact(on_gpu * (1.0 + 1e-6), on_cpu, "GPU 上改坏")
