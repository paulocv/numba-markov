# Optimized Markov Chain approach for epidemic spread in multiplex networks

An optimized python implementation of discrete-time Microscopic Markov Chain Approach (MMCA) [1] for epidemic models in static single-layer and multiplex networks.

The code uses numba (v 0.53.0) and awkward (1.0.0) to wrap the core calculations into low-level code, making the performance and memory usage comparable to C/Fortran.

**Disclaimer:**  this is not a python package. Currently this repository is meant to display the code that I use in my research project. Upon enough request, I can consider turning it into an installable package. 

_This repository is a work in progress!!_

## References

[1] Gómez, Sergio, et al. "Discrete-time Markov chain approach to contact-based disease spreading in complex networks." *EPL (Europhysics Letters)* 89.3 (2010): 38009.

