"""
    PointProcesses

A package for temporal point process modeling, simulation and inference.
"""
module PointProcesses

# Imports

using DensityInterface: DensityInterface, HasDensity, densityof, logdensityof
using Distributions: Distributions, UnivariateDistribution, MultivariateDistribution
using Distributions: Categorical, Exponential, Poisson, Uniform, Dirac
using Distributions: fit, suffstats, probs, mean, support, pdf
using LinearAlgebra: dot, diagm
using Random: rand
using Random: AbstractRNG, default_rng
using StatsAPI: StatsAPI, fit

## Hidden names

# Exports

## Reexports

export logdensityof, densityof # DensityInterface
export fit # StatsAPI
export fit_map
export convert

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

### Poisson processes
export PoissonProcess
export UnivariatePoissonProcess
export MultivariatePoissonProcess, MultivariatePoissonProcessPrior

### Hawkes processes
export HawkesProcess
export UnivariateHawkesProcess, UnmarkedUnivariateHawkesProcess
export MultivariateHawkesProcess

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
include("hawkes/simulation.jl")
include("hawkes/intensity.jl")
include("hawkes/fit.jl")
include("hawkes/unmarked_univariate/unmarked_univariate_hawkes.jl")
include("hawkes/unmarked_univariate/intensity.jl")
include("hawkes/unmarked_univariate/time_change.jl")
include("hawkes/unmarked_univariate/fit.jl")
include("hawkes/unmarked_univariate/simulation.jl")
include("hawkes/univariate/univariate_hawkes.jl")
include("hawkes/univariate/intensity.jl")
include("hawkes/univariate/time_change.jl")
include("hawkes/univariate/fit.jl")
include("hawkes/univariate/simulation.jl")
include("hawkes/multivariate/multivariate_hawkes.jl")
include("hawkes/multivariate/intensity.jl")
# include("hawkes/multivariate/time_change.jl")
# include("hawkes/multivariate/fit.jl")
include("hawkes/multivariate/simulation.jl")

end
