# Beyond the Dimensions: A Structured Evaluation of Multivariate Time Series Distance Measures
Code and analysis associated with the paper "Beyond the Dimensions: A Structured Evaluation of Multivariate Time Series Distance Measures", submitted to MulTiSa 24' (Utrecht).
The research was aimed at performing a structured, large-scale evaluation of distance measures for multivariate time series (MTS).

# Folder structure
- `src/`: Contains the source code of the algorithm, all related baselines, and analysis notebooks for the experiments.
- `src/measures`: Contains the implementations of the distance measures, as well as the abstract class for the measures.
- `src/classification_test.py`: The main script for the classification experiment.

# Instructions for code
All measures are implemented through an abstract class in `src/model.py`, with the concrete implementations in `src/measures`.
As one will notice, all measures include a `<measure>_all` method, which computes the (full) distance matrix between all pairs of time series in a dataset.
The distance computation itself generally involves some simple matrix operations through numpy, optionally with some help of dedicated libraries such as hmmlearn.
The code was implemented to be simple and easy to understand, and is therefore not optimized for speed.
Optimization of the code through parallelization or implementation in a faster language (e.g. Java/Matlab) might be necessary for the full paper.

# Authors
- Jens E. d'Hondt, Eindhoven University of Technology ([email](mailto:j.e.d.hondt@tue.nl))

