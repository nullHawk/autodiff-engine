import numpy as np

from autodiff import Value


def _finite_diff_grad(f, x, eps=1e-6):
    """Finite-difference gradient for scalar-output function f(x)."""
    x = np.array(x, dtype=float, copy=True)
    grad = np.zeros_like(x, dtype=float)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]

        x[idx] = old + eps
        f_pos = float(f(x))

        x[idx] = old - eps
        f_neg = float(f(x))

        x[idx] = old
        grad[idx] = (f_pos - f_neg) / (2.0 * eps)
        it.iternext()

    return grad


def test_scalar_chain_rule_and_accumulation():
    x = Value(3.0)
    y = x * x + x  # y = x^2 + x
    y.backward()

    assert np.allclose(y.data, 12.0)
    assert np.allclose(x.grad, 7.0)  # dy/dx = 2x + 1 at x=3


def test_broadcast_add_mul_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    a0 = rng.normal(size=(2, 1))
    b0 = rng.normal(size=(1, 3))

    def loss_from_a(a):
        aV = Value(a)
        bV = Value(b0)
        out = ((aV + bV) * (aV + 2.0)).sum()
        return out.data

    def loss_from_b(b):
        aV = Value(a0)
        bV = Value(b)
        out = ((aV + bV) * (aV + 2.0)).sum()
        return out.data

    a = Value(a0)
    b = Value(b0)
    out = ((a + b) * (a + 2.0)).sum()
    out.backward()

    g_a_fd = _finite_diff_grad(loss_from_a, a0)
    g_b_fd = _finite_diff_grad(loss_from_b, b0)

    assert a.grad.shape == a0.shape
    assert b.grad.shape == b0.shape

    assert np.allclose(a.grad, g_a_fd, rtol=1e-5, atol=1e-6)
    assert np.allclose(b.grad, g_b_fd, rtol=1e-5, atol=1e-6)


def test_tanh_exp_log_backward_shapes_and_values():
    x0 = np.array([[0.1, -0.2], [0.3, 0.05]])

    x = Value(x0)
    out = (x.tanh().exp() + (x + 2.5).log()).sum()
    out.backward()

    # Finite-diff check
    def loss(xx):
        xxV = Value(xx)
        return (xxV.tanh().exp() + (xxV + 2.5).log()).sum().data

    g_fd = _finite_diff_grad(loss, x0)
    assert x.grad.shape == x0.shape
    assert np.allclose(x.grad, g_fd, rtol=1e-5, atol=1e-6)


def test_sum_axis_keepdims_backward():
    x0 = np.arange(24, dtype=float).reshape(2, 3, 4) / 10.0
    x = Value(x0)

    out = x.sum(axis=1, keepdims=False).sum()
    out.backward()

    assert x.grad.shape == x0.shape
    assert np.allclose(x.grad, np.ones_like(x0))


def test_batched_matmul_backward_matches_finite_differences_small():
    rng = np.random.default_rng(1)
    a0 = rng.normal(size=(2, 2, 3))
    b0 = rng.normal(size=(2, 3, 2))

    def loss_from_a(a):
        aV = Value(a)
        bV = Value(b0)
        return (aV @ bV).sum().data

    def loss_from_b(b):
        aV = Value(a0)
        bV = Value(b)
        return (aV @ bV).sum().data

    a = Value(a0)
    b = Value(b0)
    out = (a @ b).sum()
    out.backward()

    g_a_fd = _finite_diff_grad(loss_from_a, a0)
    g_b_fd = _finite_diff_grad(loss_from_b, b0)

    assert np.allclose(a.grad, g_a_fd, rtol=1e-5, atol=1e-6)
    assert np.allclose(b.grad, g_b_fd, rtol=1e-5, atol=1e-6)
