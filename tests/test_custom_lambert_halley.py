"""custom_lambert_halley: W0 accuracy over the whole float range, step count, and torch behaviour."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import lambertw as scipy_lambertw
import torch

import custom_lambert_halley as clh


def _float64_range():
    log_z = np.concatenate([[-745.13], np.linspace(-744.4, 709.78, 100001)])
    return torch.cat([torch.tensor([0.0, 5e-324], dtype=torch.float64), torch.tensor(np.exp(log_z)),
                      torch.tensor([np.finfo(np.float64).max], dtype=torch.float64)])


def test_matches_scipy_from_zero_to_the_float64_maximum():
    z = _float64_range()
    w = clh.lambertw(z).numpy()
    expected = scipy_lambertw(z.numpy()).real
    nonzero = expected != 0
    np.testing.assert_allclose(w[nonzero], expected[nonzero], rtol=4e-16, atol=0)
    assert np.array_equal(w[~nonzero], expected[~nonzero])
    # Denormals: W(z) = z to the last bit.
    assert clh.lambertw(torch.tensor([5e-324, 1e-310], dtype=torch.float64)).tolist() == [5e-324, 1e-310]


def test_every_value_converges_within_four_steps():
    z = _float64_range()
    torch.testing.assert_close(clh.lambertw(z, max_steps=4), clh.lambertw(z, max_steps=12), rtol=0, atol=0)
    assert not torch.equal(clh.lambertw(z, max_steps=2), clh.lambertw(z))


def test_float32_is_accurate_to_its_roundoff():
    z = torch.tensor(np.exp(np.linspace(-80, 88.7, 20001)), dtype=torch.float32)
    w = clh.lambertw(z)
    assert w.dtype == torch.float32
    expected = scipy_lambertw(z.double().numpy()).real
    np.testing.assert_allclose(w.double().numpy(), expected, rtol=2.5e-7, atol=0)


def test_special_values_and_domain():
    z = torch.tensor([-1.0, -1e-300, 0.0, float("inf"), float("nan")], dtype=torch.float64)
    w = clh.lambertw(z)
    assert torch.isnan(w[0]) and torch.isnan(w[1]) and torch.isnan(w[4])
    assert w[2] == 0.0 and w[3] == float("inf")
    with pytest.raises(TypeError):
        clh.lambertw(torch.tensor([1, 2, 3]))


def test_keeps_shape_and_satisfies_the_defining_equation():
    z = torch.rand(4, 3, 5, dtype=torch.float64) * 1e3
    w = clh.lambertw(z)
    assert w.shape == z.shape
    torch.testing.assert_close(w * torch.exp(w), z, rtol=4e-15, atol=0)


def test_compiles_as_one_graph():
    z = torch.tensor(np.exp(np.linspace(-30, 80, 513)), dtype=torch.float64)
    compiled = torch.compile(clh.lambertw, fullgraph=True)
    torch.testing.assert_close(compiled(z), clh.lambertw(z), rtol=4e-16, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu():
    z = _float64_range()
    torch.testing.assert_close(clh.lambertw(z.cuda()).cpu(), clh.lambertw(z), rtol=4e-16, atol=0)
