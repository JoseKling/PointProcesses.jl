"""
    PoissonProcess{R,D}

Homogeneous temporal Poisson process with arbitrary mark distribution.

# Fields

- `λ::R`: ground intensity.
- `mark_dist::D`: mark distribution.

# Constructor

    PoissonProcess(λ, mark_dist)
"""
struct PoissonProcess{R<:Real,D} <: AbstractPointProcess
    λ::R
    mark_dist::D

    function PoissonProcess(λ::R, mark_dist::D; check_args::Bool=true) where {R<:Real,D}
        check_args &&
            λ < zero(λ) &&
            throw(
                DomainError(
                    "λ = $λ", "PoissonProcess: the ground intensity λ must be non negative."
                ),
            )
        return new{R,D}(λ, mark_dist)
    end
end

function Base.show(io::IO, pp::PoissonProcess)
    return print(io, "PoissonProcess($(pp.λ), $(pp.mark_dist))")
end

## Alias 
"""
    UnivariatePoissonProcess{R}

Homogeneous univariate temporal Poisson process with scalar intensity `λ::R`.

`UnivariatePoissonProcess{R}` is simply a type alias for `PoissonProcess{R,Dirac{Nothing}}`.
"""
const UnivariatePoissonProcess{R<:Real} = PoissonProcess{R,Dirac{Nothing}}

function Base.show(io::IO, pp::UnivariatePoissonProcess)
    return print(io, "UnivariatePoissonProcess($(pp.λ))")
end

"""
    MultivariatePoissonProcess{R}

Homogeneous multivariate temporal Poisson process with marginal intensities of type `R`.

`MultivariatePoissonProcess{R}` is simply a type alias for `PoissonProcess{R,Categorical{Float64,Vector{Float64}}}`.
"""
# The choice to impose the mark distribution Categorical{Float64,Vector{Float64}} was made on purpose
## It is mainly due to the fact that Distributions.Categorical makes (most of the time) automatic conversions to Float64
const MultivariatePoissonProcess{R<:Real} = PoissonProcess{
    R,Categorical{Float64,Vector{Float64}}
}

function Base.show(io::IO, pp::MultivariatePoissonProcess)
    return print(io, "MultivariatePoissonProcess($(pp.λ * probs(pp.mark_dist)))")
end

## Constructors
function PoissonProcess(λ::Vector{R}; check_args::Bool=true) where {R<:Real}
    if check_args
        if any(λ .< zero(λ))
            throw(
                DomainError(
                    "λ = $λ",
                    "PoissonProcess: the condition λ ≥ 0 is not satisfied for all dimensions.",
                ),
            )
        end
        if sum(λ) == 0
            return PoissonProcess(
                0.0, Categorical(ones(length(λ)) / length(λ)); check_args=check_args
            )
        end
    end
    return PoissonProcess(sum(λ), Categorical(λ / sum(λ)); check_args=check_args)
end

function PoissonProcess(λ::R; check_args::Bool=true) where {R<:Real}
    return PoissonProcess(λ, Dirac(nothing); check_args=check_args)
end
PoissonProcess() = PoissonProcess(1.0)

## Access
ground_intensity(pp::PoissonProcess) = pp.λ
mark_distribution(pp::PoissonProcess) = pp.mark_dist

"""
    intensity_vector(pp<:MultivariatePoissonProcess)

Compute the vector of the marginal intensities `λ` for a multivariate Poisson process.
"""
function intensity_vector(pp::MultivariatePoissonProcess{R}) where {R}
    return ground_intensity(pp) .* probs(mark_distribution(pp))
end

## Intensity functions
function intensity(pp::PoissonProcess, m)
    return ground_intensity(pp) * densityof(mark_distribution(pp), m)
end

function log_intensity(pp::PoissonProcess, m)
    return log(ground_intensity(pp)) + logdensityof(mark_distribution(pp), m)
end

## Implementing AbstractPointProcess interface

ground_intensity(pp::PoissonProcess, t, h) = ground_intensity(pp)
mark_distribution(pp::PoissonProcess, t, h) = mark_distribution(pp)
mark_distribution(pp::PoissonProcess, t) = mark_distribution(pp) # For simulate_ogata
intensity(pp::PoissonProcess, m, t, h) = intensity(pp, m)
log_intensity(pp::PoissonProcess, m, t, h) = log_intensity(pp, m)

function ground_intensity_bound(pp::PoissonProcess, t::T, h) where {T<:Real}
    B = ground_intensity(pp)
    L = typemax(T)
    return (B, L)
end

function integrated_ground_intensity(pp::PoissonProcess, h, a, b)
    return ground_intensity(pp) * (b - a)
end
