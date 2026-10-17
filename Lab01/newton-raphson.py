import numpy as np


# f(x, y) = x**4 + y**4 + 2*x**2*y
def grad(u):
    x, y = u
    return np.array([4 * x**3 + 4 * x * y, 4 * y**3 + 2 * x**2])


def hessian(u):
    x, y = u
    return np.array([[12 * x**2 + 4 * y, 4 * x], [4 * x, 12 * y**2]])


def newton_optim(u_0, grad_fn, hessian_fn, epsilon=1e-8, max_iter=100):
    u = np.array(u_0, dtype=float)
    for _ in range(max_iter):
        grad_u = grad_fn(u)

        if np.linalg.norm(grad_u) < epsilon:
            break

        hessian_u = hessian_fn(u)

        t = np.linalg.solve(hessian_u, grad_u)
        u = u - t

    return u


if __name__ == "__main__":
    u_0 = (10, 10)
    u_optim = newton_optim(u_0, grad, hessian, epsilon=1e-8, max_iter=100)

    print(u_optim)