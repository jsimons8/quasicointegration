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


```
quasicointegration 1 /Users/jerome/Documents/near-cointegration/empirical-applications-data/term-spread/termspread.csv GS1 GS10 /Users/jerome/CLionProjects/data/weights.csv /Users/jerome/CLionProjects/data/adjusted_critical_value.csv 19 8 25
quasicointegration 2 /Users/jerome/Documents/near-cointegration/empirical-applications-data/term-spread/termspread.csv GS1 GS10 /Users/jerome/CLionProjects/data/weights.csv 19 8 25 /Users/jerome/Documents/near-cointegration/empirical 19 8 25 0.05 100
```

## Compilation and installation

### System requirements

To run this software, you need `cmake`, `gcc`, `ninja` all of which are free software.

The code archive supports the standard gnu toolchain with `ninja` as the recommended make programme. To compile, clone this repository and change directory into the folder `quasicointegration`.  

~~~~
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM=ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++-12 -G Ninja quasicointegration
~~~~

Then, compile with the command:

```
cmake --build ./cmake-build-debug --target quasicointegration
```

## Future work

### Interfacing with C libraries

https://eigen.tuxfamily.org/dox/classEigen_1_1Map.html

https://forum.kde.org/viewtopic.php?f=74&t=127738

https://forum.kde.org/viewtopic.php?f=74&t=143476

https://stackoverflow.com/questions/17036818/initialise-eigenvector-with-stdvector



### Higher order models


For higher order models, we need some trick for numerical integration or alternatively, if we want to
keep a grid, then some log sum exp trick.

https://discourse.julialang.org/t/numerical-integration-only-with-evaluated-point/63552

https://discourse.julialang.org/t/romberg-jl-simple-romberg-integration-in-julia/39313

https://github.com/dextorious/NumericalIntegration.jl

https://theochem.github.io/horton/2.1.1/lib/mod_horton_grid_atgrid.html


