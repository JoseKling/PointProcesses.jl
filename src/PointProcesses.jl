"""
    PointProcesses

A package for temporal point process modeling, simulation and inference.
"""
module PointProcesses

# Imports

using DensityInterface: DensityInterface, HasDensity, densityof, logdensityof
using Distributions: Distributions, UnivariateDistribution, MultivariateDistribution
using Distributions: Categorical, Exponential, Poisson, Uniform
using Distributions: fit, suffstats
using LinearAlgebra: dot
using Random: rand
using Random: AbstractRNG, default_rng
using StatsAPI: StatsAPI, fit, HypothesisTest, pvalue
using HypothesisTests: ExactOneSampleKSTest, ksstats

## Hidden names

# Exports

## Reexports

export logdensityof, densityof # DensityInterface
export fit # StatsAPI
export fit_map
export HypothesisTest, pvalue

## History

export History
export event_times, event_marks, min_time, max_time, min_mark, max_mark
export nb_events, has_events, duration
export time_change, split_into_chunks

## Point processes

export AbstractPointProcess
export BoundedPointProcess
export ground_intensity, mark_distribution
export intensity, log_intensity
export ground_intensity_bound
export integrated_ground_intensity
export simulate_ogata

## Models

export AbstractPoissonProcess
export MultivariatePoissonProcess, MultivariatePoissonProcessPrior
export MarkedPoissonProcess

## Hypothesis testset

export Statistic, KSDistance
export BootstrapTest, NoBootstrapTest

# Includes

include("history.jl")
include("abstract_point_process.jl")
include("simulation.jl")
include("bounded_point_process.jl")

include("poisson/abstract_poisson_process.jl")
include("poisson/simulation.jl")

include("poisson/multivariate/multivariate_poisson_process.jl")
include("poisson/multivariate/suffstats.jl")
include("poisson/multivariate/prior.jl")
include("poisson/multivariate/fit.jl")

include("poisson/marked/marked_poisson_process.jl")
include("poisson/marked/fit.jl")

include("HypothesisTests/Statistics.jl")
include("HypothesisTests/Statistics/KSDistance.jl")
include("HypothesisTests/PPGoFTests/BootstrapTest.jl")
include("HypothesisTests/PPGoFTests/NoBootstrapTest.jl")
end
