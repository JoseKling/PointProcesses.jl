"""
    HawkesProcess{T<:Real,D}

Hawkes process with exponential decay kernel and mark distribution `D`.

A Hawkes process is a self-exciting point process where each event increases the probability
of future events. The conditional intensity function is given by

    λ(t) = μ + ∑_{tᵢ < t} f(mᵢ) exp(-g(mᵢ)(t - tᵢ)),

where the sum is over all previous event times (tᵢ, mᵢ).

# Fields
- `μ::T`: baseline intensity (immigration rate)
- `f`: function f:M → R, where M is the space of marks. Jump size.
- `g`: function f:M → R, where M is the space of marks. Decay rate.
"""
struct HawkesProcess{T<:Real,D} <: AbstractPointProcess
    μ::T
    f
    g
    mark_dist::D
end

function Base.show(io::IO, hp::HawkesProcess)
    mean_mark = mean(hp.mark_dist)
    return print(io, "HawkesProcess($(hp.μ), $(hp.f(mean_mark)), $(hp.g(mean_mark)), $(hp.mark_dist))")
end

## Alias 
"""
    UnivariateHawkesProcess{R}

Unmarked univariate temporal Hawkes process.

The conditional intensity function is given by

    λ(t) = μ + ∑_{tᵢ < t} α exp(-β(t - tᵢ)).

Equivalent to the general definition with f(m) = α and g(m) = ω.

# Fields
- `μ::R`: baseline intensity (immigration rate)
- `α::R`: Jump size.
- `ω::R`: Decay rate.

Alias for `HawkesProcess{R,Dirac{Nothing}}`.
"""
const UnivariateHawkesProcess{R<:Real} = HawkesProcess{R,Dirac{Nothing}}

function Base.show(io::IO, pp::UnivariateHawkesProcess)
    return print(io, "UnivariateHawkesProcess($(pp.μ), $(pp.f(nothing)), $(pp.g(nothing)))")
end

"""
    MultivariateHawkesProcess{R}

Unmarked multivariate temporal Hawkes process

For a process with D marginal processes, the conditional intensity
function of the k-th process is given by

    λₖ(t) = μₖ + ∑_{h=1,...,D} ∑_{tʰᵢ < t} αₖₕ exp(-βₖₕ(t - tʰᵢ)),

where tʰᵢ is the i-th element in the h-th marginal process.
μ is a vector of length D with elements μᵢ. α and ω are D×D
matrices with elements αᵢⱼ and ωᵢⱼ.

# Fields
- `μ::Vector{<:Real}`: baseline intensity (immigration rate)
- `α::Matrix{<:Real}`: Jump size.
- `ω::Matrix{<:Real}`: Decay rate.

Alias for `HawkesProcess{R,Categorical{Float64,Vector{Float64}}}`.
"""
const MultivariateHawkesProcess{R<:Real} = HawkesProcess{
    R,Categorical{Float64,Vector{Float64}}
}
# The choice to impose the mark distribution Categorical{Float64,Vector{Float64}} was made on purpose
## It is mainly due to the fact that Distributions.Categorical makes (most of the time) automatic conversions to Float64

function Base.show(io::IO, pp::MultivariateHawkesProcess)
    D = length(pp.mark_dist.p)
    inds = [(i, j) for i in 1:D, j in 1:D]
    return print(io, "MultivariateHawkesProcess\nμ = $(pp.μ .* pp.mark_dist.p)\nα = $(pp.f.(inds))\nω = $(pp.g.(inds))")
end

## Constructors
### UnivariatehawkesProcess
function HawkesProcess(
    μ::R1, α::R2, ω::R3; check_args::Bool=true
    ) where {R1<:Real,R2<:Real,R3<:Real}
    check_args && check_args_Hawkes(μ, α, ω)
    R = promote_type(R1, R2, R3)
    fα(_...) = R(α)
    gω(_...) = R(ω)
    return HawkesProcess(R(μ), fα, gω, Dirac(nothing))
end

### MultivariateHawkesProcess
function HawkesProcess(
    μ::Vector{R1}, α::Matrix{R2}, ω::Matrix{R3}; check_args::Bool=true
    ) where {R1<:Real,R2<:Real,R3<:Real}
    check_args && check_args_Hawkes(μ, α, ω)
    R = promote_type(R1, R2, R3)
    f((i, j)) = R(α[i, j])
    g((i, j)) = R(ω[i, j])
    return HawkesProcess(R(sum(μ)), f, g, Categorical(Float64.(μ / sum(μ))))
end

### Check check
# Univariate Hawkes
function check_args_Hawkes(μ::Real, α::Real, ω::Real)
    if any((μ, α, ω) .< 0)
        throw(
            DomainError(
                "μ = $μ, α = $α, ω = $ω",
                "HawkesProcess: All parameters must be non-negative.",
            ),
        )
    end
    return nothing
end

# Multivariate Hawkes
function check_args_Hawkes(μ::Vector{<:Real}, α::Matrix{<:Real}, ω::Matrix{<:Real})
    if !(length(μ) == size(α)[1] && length(μ) == size(α)[2])
        throw(DimansionMismatch("α must have size $(length(μ))×$(length(μ))")) 
    end
    if !(length(μ) == size(ω)[1] && length(μ) == size(ω)[2])
        throw(DimansionMismatch("ω must have size $(length(μ))×$(length(μ))"))
    end
    if any(μ .< zero(μ))
        throw(
            DomainError(
                "μ = $μ",
                "HawkesProcess: the condition μ ≥ 0 is not satisfied for all dimensions.",
            ),
        )
    end
    if any(α .< zero(α))
        throw(
            DomainError(
                "α = $α",
                "HawkesProcess: the condition α ≥ 0 is not satisfied for all dimensions.",
            ),
        )
    end
    if any(ω .< zero(ω))
        throw(
            DomainError(
                "ω = $ω",
                "HawkesProcess: the condition ω ≥ 0 is not satisfied for all dimensions.",
            ),
        )
    end
    return nothing
end

## Access
function ground_intensity(hp::HawkesProcess, h::History, t)
    activation = sum(exp.(hp.ω .* (@view h.times[1:(searchsortedfirst(h.times, t) - 1)])))
    return hp.μ + (hp.α * activation / exp(hp.ω * t))
end

function integrated_ground_intensity(hp::HawkesProcess, h::History, tmin, tmax)
    times = event_times(h, h.tmin, tmax)
    integral = 0
    for ti in times
        # Integral of activation function. 'max(tmin - ti, 0)' corrects for events that occurred
        # inside or outside the interval [tmin, tmax].
        integral += (exp(-hp.ω * max(tmin - ti, 0)) - exp(-hp.ω * (tmax - ti)))
    end
    integral *= hp.α / hp.ω
    integral += hp.μ * (tmax - tmin) # Integral of base rate
    return integral
end

function DensityInterface.logdensityof(hp::HawkesProcess, h::History)
    A = zeros(nb_events(h)) # Vector A in Ozaki (1979)
    for i in 2:nb_events(h)
        A[i] = exp(-hp.ω * (h.times[i] - h.times[i - 1])) * (1 + A[i - 1])
    end
    return sum(log.(hp.μ .+ (hp.α .* A))) - # Value of intensity at each event
           (hp.μ * duration(h)) - # Integral of base rate
           ((hp.α / hp.ω) * sum(1 .- exp.(-hp.ω .* (duration(h) .- h.times)))) # Integral of each kernel
end

function time_change(hp::HawkesProcess, h::History{T,M}) where {T<:Real,M}
    n = nb_events(h)
    A = zeros(T, n + 1) # Array A in Ozaki (1979)
    @inbounds for i in 2:n
        A[i] = exp(-hp.ω * (h.times[i] - h.times[i - 1])) * (1 + A[i - 1])
    end
    A[end] = exp(-hp.ω * (h.tmax - h.times[end])) * (1 + A[end - 1]) # Used to calculate the integral of the intensity at every event time
    times = T.(hp.μ .* (h.times .- h.tmin)) # Transformation with respect to base rate
    T_base = hp.μ * duration(h) # Contribution of base rate to total length of time re-scaled process
    for i in eachindex(times)
        times[i] += (hp.α / hp.ω) * ((i - 1) - A[i]) # Add contribution of activation functions
    end
    return History(;
        times=times,
        marks=h.marks,
        tmin=zero(T),
        tmax=T(T_base + ((hp.α / hp.ω) * (n - A[end]))),
        check=false,
    ) # A time re-scaled process starts at t=0
end

