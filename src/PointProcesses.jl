"""
    PointProcesses

A package for temporal point process modeling, simulation and inference.
"""
module PointProcesses

# Imports

using DensityInterface: DensityInterface, HasDensity, densityof, logdensityof
using Distributions: Distributions, UnivariateDistribution, MultivariateDistribution
using Distributions: Categorical, Exponential, Poisson, Uniform, Dirac
using Distributions: fit, suffstats, probs
using Integrals: Integrals, IntegralProblem, solve, QuadGKJL
using LinearAlgebra: dot
using Optim: Optim, optimize, LBFGS
using Random: rand
using Random: AbstractRNG, default_rng, Xoshiro
using StatsAPI: StatsAPI, fit, HypothesisTest, pvalue
using HypothesisTests: ExactOneSampleKSTest, ksstats

## Hidden names

# Exports

## Reexports

export logdensityof, densityof # DensityInterface
export fit # StatsAPI
export fit_map
export HypothesisTest, pvalue # HypothesisTests

## History

export History
export event_times, event_marks, event_dims
export min_time, max_time, min_mark, max_mark
export nb_events, has_events, duration, ndims
export time_change, split_into_chunks

## Point processes

export AbstractPointProcess, AbstractUnivariateProcess, AbstractMultivariateProcess
export BoundedPointProcess
export ground_intensity, mark_distribution
export intensity, log_intensity
export ground_intensity_bound
export integrated_ground_intensity
export simulate_ogata, simulate

## Models

export PoissonProcess
export UnivariatePoissonProcess
export MultivariatePoissonProcess, MultivariatePoissonProcessPrior
export InhomogeneousPoissonProcess
export HawkesProcess
export IndependentMultivariateProcess

## Intensity functions for inhomogeneous processes

export PolynomialIntensity
export ExponentialIntensity
export SinusoidalIntensity
export PiecewiseConstantIntensity
export LinearCovariateIntensity

## Parametric intensity traits and configuration

export ParametricIntensity
export from_params
export IntegrationConfig

## Goodness of fit tests tests

export Statistic, KSDistance, statistic
export PointProcessTest, BootstrapTest, MonteCarloTest

# Includes

## General
include("history.jl")
include("abstract_point_process.jl")
include("simulation.jl")
include("bounded_point_process.jl")

## Univariate processes

### Homogeneous Poisson
include("univariate/poisson/poisson_process.jl")
include("univariate/poisson/suffstats.jl")
include("univariate/poisson/prior.jl")
include("univariate/poisson/fit.jl")
include("univariate/poisson/simulation.jl")

### Inhomogeneous Poisson
include("univariate/poisson/inhomogeneous/integration_config.jl")
include("univariate/poisson/inhomogeneous/parametric_intensity.jl")
include("univariate/poisson/inhomogeneous/intensity_functions.jl")
include("univariate/poisson/inhomogeneous/intensity_methods.jl")
include("univariate/poisson/inhomogeneous/inhomogeneous_poisson_process.jl")
include("univariate/poisson/inhomogeneous/fit.jl")

### Hawkes
include("univariate/hawkes/hawkes_process.jl")

## Multivariate processes

### Independent multivariate processes
include("multivariate/independent_multivariate.jl")

### Multivariate Poisson processes
include("multivariate/multivariate_poisson_process.jl")

## Hypothesis tests
include("HypothesisTests/point_process_tests.jl")
include("HypothesisTests/statistic.jl")
include("HypothesisTests/Statistics/KSDistance.jl")
include("HypothesisTests/PPTests/bootstrap_test.jl")
include("HypothesisTests/PPTests/monte_carlo_test.jl")
end
