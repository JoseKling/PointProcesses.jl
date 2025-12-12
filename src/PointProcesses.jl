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
using LinearAlgebra: dot
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
export event_times, event_marks, min_time, max_time, min_mark, max_mark
export nb_events, has_events, duration
export time_change, split_into_chunks

## Point processes

export AbstractPointProcess
export BoundedPointProcess
export ground_intensity, mark_distribution
export intensity, log_intensity, intensity_vector
export ground_intensity_bound
export integrated_ground_intensity
export simulate_ogata, simulate

## Models

export PoissonProcess
export UnivariatePoissonProcess
export MultivariatePoissonProcess, MultivariatePoissonProcessPrior
export HawkesProcess

## Goodness of fit tests tests

export Statistic, KSDistance, statistic
export PointProcessTest, BootstrapTest, MonteCarloTest

# Includes

include("history.jl")
include("abstract_point_process.jl")
include("simulation.jl")
include("bounded_point_process.jl")

include("poisson/poisson_process.jl")
include("poisson/suffstats.jl")
include("poisson/prior.jl")
include("poisson/fit.jl")
include("poisson/simulation.jl")

include("hawkes/hawkes_process.jl")

include("HypothesisTests/point_process_tests.jl")
include("HypothesisTests/statistic.jl")
include("HypothesisTests/Statistics/KSDistance.jl")
include("HypothesisTests/PPTests/bootstrap_test.jl")
include("HypothesisTests/PPTests/monte_carlo_test.jl")
end
