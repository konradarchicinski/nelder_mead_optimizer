from nelder_mead_optimizer import nelder_mead

if __name__ == "__main__":

    def styblinski_tang_function(x):
        return 0.5 * sum([xi**4 - 16*xi**2 + 5*xi for xi in x])

    print(nelder_mead(
        styblinski_tang_function, [0., 0.], 0.1, 10e-6, 10, 100, 1.0, 2.0, -0.5, 0.5)
    )