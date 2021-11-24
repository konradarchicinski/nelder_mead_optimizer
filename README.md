# Nelder-Mead optimizer

Pure Rust implementation of the Nelder-Mead optimization algorithm with Python binding, based od PyO3 library.

Python Installation
------------

*(tested on Windows with Python 3.8 and Rust 1.56)*

 - clone this repository by `git clone https://github.com/konradarchicinski/nelder_mead_optimizer`
 - activate Python virtual enviroment and install maturin with `pip install maturin`
 - run `maturin develop --release` from the folder where the repository was downloaded
 - after a successful compilation the library should be installed in a virtual environment

## References

*Nelder-Mead algorithm:* https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method <br>
*Python code equivalent:* https://github.com/fchollet/nelder-mead <br>