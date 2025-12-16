
# Reverse‑Mode Autodiff Engine

This repo implements a small **reverse‑mode automatic differentiation** engine (a.k.a. backprop) in pure Python.

Core idea: every operation creates a node in a computation graph (a **DAG**: directed acyclic graph), and calling `.backward()` on the final output runs reverse‑mode backprop to populate `.grad` on all dependent inputs.

Internally, `.backward()` performs a **topological sort** over this DAG so gradients are propagated in a valid reverse order (from outputs back to inputs).

## Features

- `Value` scalar **and** N‑D tensor support (internally uses NumPy arrays)
- Reverse‑mode backprop with topological sort
- Broadcasting-aware gradients for elementwise ops
- Batched matrix multiplication gradients via `@`
- A notebook (`autodiff_scaler.ipynb`) that demos the concepts and includes simple performance plots

## Benchmarks
<img width="625" height="398" alt="download" src="https://github.com/user-attachments/assets/b203f3d8-ddc6-4ca0-8af2-4878c841f4cf" />
<img width="657" height="393" alt="download" src="https://github.com/user-attachments/assets/e4698dbf-1ba2-47b9-a999-79bc8712d613" />


## Project layout

```
autodiff/
	__init__.py      # exports Value
	value.py         # Value implementation
tests/
	test_value.py    # unit tests
autodiff_scaler.ipynb
```

## Installation

This project depends on **NumPy**.

```bash
python -m pip install -U numpy
```

If you want to run the tests using `pytest`:

```bash
python -m pip install -U pytest
```

## Quick start

```python
import numpy as np
from autodiff import Value

# Scalars
x = Value(3.0)
y = x * x + x          # y = x^2 + x
y.backward()
print(y.data)          # 12.0
print(x.grad)          # 7.0 (since dy/dx = 2x + 1)

# Tensors (N-D)
a = Value(np.random.randn(2, 3))
b = Value(np.random.randn(1, 3))
loss = ((a + b).tanh() * (a + 2.0)).sum()  # broadcasting works
loss.backward()
print(a.grad.shape)    # (2, 3)
print(b.grad.shape)    # (1, 3)
```

## Supported ops (current)

- Elementwise: `+`, `*`, unary `-`, `-`, `/`, `**` (scalar exponent)
- Activations / unary: `.tanh()`, `.exp()`, `.log()`
- Reductions: `.sum(axis=None, keepdims=False)`
- Matrix multiply: `@` for 2D and batched matmul for N‑D (`(..., m, k) @ (..., k, n)`)

Gradients are accumulated into `.grad` (same shape as `.data`).


## Running tests

From the repo root:

```bash
pytest -q
```

If you prefer not to use `pytest`, you can still run the test file as a script, but you’ll get the best output with `pytest`.

## Notes / next steps

Typical next features for a tiny tensor autograd engine:

- `relu`, `sigmoid`, `softmax` (with stable log-sum-exp)
- `mean`, `reshape`, `transpose`, slicing/indexing
- parameter containers (layers) and an optimizer loop
- Implementing core tensor computations (matmuls etc.) in `CBLAS` or `CuBLAS` for optimization.

