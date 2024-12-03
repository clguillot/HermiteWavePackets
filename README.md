HermiteWavePackets is a Julia package meant to handle elementary computations on several types of Hermite and Gaussian type functions.
As to this day, the package includes the support for the following types of functions:

$$
  P(x) e^{-\frac a 2 (x - q)^2}
$$

for $a>0$, $q \in \mathbb{R}$, and $P$ a polynomial

$$
  P(x) e^{-\frac z 2 (x - q)^2} e^{ipx}
$$

for $z \in \mathbb C$, $q, p \in \mathbb R$, and $P$ a polynomial
