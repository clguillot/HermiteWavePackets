# HermiteWavePackets.jl

**HermiteWavePackets.jl** is a Julia package designed to handle elementary computations involving Hermite and Gaussian-type functions. The package supports the following classes of functions:

1. **Real Gaussian Polynomials**  
   Functions of the form:
   ```math
   P(x) e^{-\frac{a}{2} (x - q)^2},
   ``` 
   where:
   - $`a \in \mathbb R_+^*`$,
   - $`q \in \mathbb R`$,
   - $`P(x)`$ is a polynomial.

2. **Complex Gaussian Wave Packets**  
   Functions of the form:  
   ```math
   P(x) e^{-\frac{z}{2} (x - q)^2} e^{i p x},
   ```
   where:
   - $`z \in \mathbb C`$ is a complex parameter (typically with \(\text{Re}(z) > 0\) to ensure decay),
   - $`q, p \in \mathbb{R}`$ are real-valued parameters representing spatial shift and momentum, respectively,
   - $`P(x)`$ is a polynomial.

The package is designed to provide efficient and accurate tools for computations involving these functions, which appear frequently in quantum mechanics, signal processing, and mathematical analysis.
