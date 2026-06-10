import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import time


# Part א - solve Ax=b using scipy instead of numpy

def scipy_solve(a, b):
    """
    Solves the linear equation Ax = b using scipy.optimize.root.

    a - the coefficient matrix (n x n)
    b - the right-hand side vector (n)
    returns x such that a @ x = b

    >>> import numpy as np
    >>> a = np.array([[2.0, 1.0], [1.0, 3.0]])
    >>> b = np.array([5.0, 10.0])
    >>> x = scipy_solve(a, b)
    >>> np.allclose(a @ x, b)
    True
    """
    # define f(x) = Ax - b, we want to find x where f(x) = 0
    def equations(x):
        return a @ x - b

    # start the search from x = [0, 0, ..., 0]
    x0 = np.zeros(len(b))
    result = scipy.optimize.root(equations, x0)
    return result.x


# Part ב - test our function against numpy on random inputs

def test_scipy_solve(num_tests=20, max_size=50):
    """
    Runs num_tests random tests and compares scipy_solve to numpy.linalg.solve.
    Prints how many passed.
    """
    passed = 0
    for _ in range(num_tests):
        # pick a random size
        n = np.random.randint(1, max_size + 1)

        # generate a random matrix and vector
        a = np.random.randn(n, n)
        b = np.random.randn(n)

        x_numpy = np.linalg.solve(a, b)
        x_scipy = scipy_solve(a, b)

        # check if both answers are close enough
        if np.allclose(x_numpy, x_scipy, atol=1e-6):
            passed += 1

    print(f"Test results: {passed}/{num_tests} passed.")
    return passed == num_tests


# Part ג - compare performance of both methods and plot the results

def compare_performance(sizes=None, repeats=3):
    """
    Measures average runtime of scipy_solve vs numpy.linalg.solve
    for matrix sizes from 1 to 1000.
    repeats - how many times to run each size (we take the average)
    """
    if sizes is None:
        # test sizes 1 to 50, then jump by 50 up to 1000
        sizes = list(range(1, 51)) + list(range(50, 1001, 50))

    numpy_times = []
    scipy_times = []

    for n in sizes:
        numpy_total = 0.0
        scipy_total = 0.0

        for _ in range(repeats):
            a = np.random.randn(n, n)
            b = np.random.randn(n)

            # time numpy
            t0 = time.perf_counter()
            np.linalg.solve(a, b)
            numpy_total += time.perf_counter() - t0

            # time scipy
            t0 = time.perf_counter()
            scipy_solve(a, b)
            scipy_total += time.perf_counter() - t0

        # save the average time
        numpy_times.append(numpy_total / repeats)
        scipy_times.append(scipy_total / repeats)

        print(f"n={n:4d} | numpy: {numpy_times[-1]:.6f}s | scipy: {scipy_times[-1]:.6f}s")

    return sizes, numpy_times, scipy_times


def plot_comparison(sizes, numpy_times, scipy_times, output_file="comparison.png"):
    """
    Plots the runtime of both solvers as a function of input size and saves the image.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, numpy_times, label="numpy.linalg.solve", color="steelblue", linewidth=2)
    plt.plot(sizes, scipy_times, label="scipy.optimize.root", color="tomato", linewidth=2)

    plt.xlabel("Matrix size (n x n)")
    plt.ylabel("Average runtime (seconds)")
    plt.title("Runtime Comparison: numpy.linalg.solve vs scipy.optimize.root")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f"Graph saved to '{output_file}'")


if __name__ == "__main__":
    import doctest

    print("=== Doctest ===")
    doctest.testmod(verbose=False)
    print("Doctests passed.\n")

    print("=== Part B: Testing scipy_solve ===")
    test_scipy_solve(num_tests=30)
    print()

    print("=== Part C: Benchmarking ===")
    sizes, numpy_times, scipy_times = compare_performance()
    print()

    print("=== Plotting ===")
    plot_comparison(sizes, numpy_times, scipy_times)
