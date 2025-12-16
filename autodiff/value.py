import numpy as np

class Value:
    def __init__(self, data, _children=None, _op="", label=""):
        # Always store as numpy array (scalars become 0-d arrays)
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data, dtype=float)

        self._backward = lambda: None
        self._prev = set(_children) if _children is not None else set()
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(shape={self.data.shape}, data={self.data})"

    @staticmethod
    def _ensure_value(x):
        return x if isinstance(x, Value) else Value(x)

    @staticmethod
    def _unbroadcast(grad, target_shape):
        """
        Reduce grad so it matches target_shape (reverse numpy broadcasting).
        """
        g = grad

        # Sum away leading dims added by broadcasting
        while g.ndim > len(target_shape):
            g = g.sum(axis=0)

        # Sum across axes where target had size 1 (broadcasted)
        for i, (gs, ts) in enumerate(zip(g.shape, target_shape)):
            if ts == 1 and gs != 1:
                g = g.sum(axis=i, keepdims=True)

        return g.reshape(target_shape)

    # -------- elementwise ops (broadcasting supported) --------

    def __add__(self, other):
        other = Value._ensure_value(other)
        out = Value(self.data + other.data, {self, other}, "+")

        def _backward():
            self.grad += Value._unbroadcast(out.grad, self.data.shape)
            other.grad += Value._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Value._ensure_value(other)
        out = Value(self.data * other.data, {self, other}, "*")

        def _backward():
            self.grad += Value._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += Value._unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, other):
        # keep this simple: scalar exponent only
        if not isinstance(other, (int, float)):
            raise TypeError("only supporting int/float powers for now")
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1.0))) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    # -------- reductions --------

    def sum(self, axis=None, keepdims=False):
        out = Value(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                # normalize negative axes
                axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                for ax in sorted(axes):
                    g = np.expand_dims(g, axis=ax)
            self.grad += np.ones_like(self.data) * g

        out._backward = _backward
        return out

    # -------- batched matmul (N-D) --------

    def __matmul__(self, other):
        other = Value._ensure_value(other)
        out = Value(self.data @ other.data, {self, other}, "@")

        def _backward():
            # For batched matmul:
            # dA = dC @ B^T, dB = A^T @ dC (with transpose on last two dims)
            bT = np.swapaxes(other.data, -1, -2)
            aT = np.swapaxes(self.data, -1, -2)

            dA = out.grad @ bT
            dB = aT @ out.grad

            # Handle any broadcasting in leading batch dims
            self.grad += Value._unbroadcast(dA, self.data.shape)
            other.grad += Value._unbroadcast(dB, other.data.shape)

        out._backward = _backward
        return out

    # -------- autograd plumbing --------

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data, dtype=float)

        for node in reversed(topo):
            node._backward()

    # -------- convenience operators --------

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return Value._ensure_value(other) + self

    def __sub__(self, other):
        return self + (-Value._ensure_value(other))

    def __rsub__(self, other):
        return Value._ensure_value(other) + (-self)

    def __rmul__(self, other):
        return Value._ensure_value(other) * self

    def __truediv__(self, other):
        other = Value._ensure_value(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return Value._ensure_value(other) * (self ** -1)