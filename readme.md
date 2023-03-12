# Empirical quasi-cointegration

The present software is the implementation of the procedure developed in Duffy and Simons (2020). For details see https://arxiv.org/abs/2002.08092.

The programme computes the cointegrating coefficient in a quasi-cointegrated vector autoregression. The procedure implements the method of Elliott, Müller, Watson (2015)

## How to use

Currently, the software supports two modes:

1. Grid-estimation of quasi-cointegrating coefficient with marginal interval.

   This mode computes the quasi-cointegrating coefficients along a grid of near-unit roots.
   
   A critical value and weights table need to be user-supplied.

2. Grid-estimation with bootstrapping for size control.
   This mode estimates the DGP and runs Monte Carlo simulations to determine the critical value to control size at the estimated parameter.

   The weights table supplied in Elliott et al. (2015) is used by default.

## Commands

`quasicointegration 2 ../data/termspread.csv GS1 GS10 ../data/weights.csv 10 8 40 ../output/ 0.05 0.0 0.03 3.841459 1000`


## Compilation and installation

### System requirements

To run this software, you need `cmake`, `gcc`, `ninja` all of which are free software.

The code archive supports the standard gnu toolchain with `ninja` as the recommended make programme. To compile, clone this repository and change directory into the folder `quasicointegration`.  

~~~~
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM=ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++-12 -G Ninja quasicointegration
~~~~

Then, compile with

```
cmake --build ./cmake-build-debug --target quasicointegration
```