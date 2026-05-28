"""
    HawkesProcess{T<:Real}

Univariate Marked Hawkes process with exponential decay kernel and events independent of marks.

A Hawkes process is a self-exciting point process where each event increases the probability
of future events. The conditional intensity function is given by:

    λ(t) = μ + α ∑_{tᵢ < t} exp(-ω(t - tᵢ))

where the sum is over all previous event times tᵢ.

# Fields

- `μ::T`: baseline intensity (immigration rate)
- `α::T`: jump size (immediate increase in intensity after an event)  
- `ω::T`: decay rate (how quickly the excitement fades)

Conditions:
- μ, α, ω >= 0
- ψ = α/ω < 1 → Stability condition. ψ is the expected number of events each event generates

Following the notation from [Lewis2011](@cite).
"""
struct HawkesProcess{T<:Real,D<:PointProcessMarkDistribution} <: AbstractUnivariateProcess
    μ::T
    α::T
    ω::T
    mark_dist::D

    function HawkesProcess(μ::T1, α::T2, ω::T3, mark_dist::D) where {T1,T2,T3<:Real,D<:PointProcessMarkDistribution}
        any((μ, α, ω) .< 0) &&
            throw(DomainError((μ, α, ω), "All parameters must be non-negative."))
        (α > 0 && α >= ω) &&
            throw(DomainError((α, ω), "Parameter ω must be strictly smaller than α"))
        T = promote_type(T1, T2, T3)
        (μ_T, α_T, ω_T) = convert.(T, (μ, α, ω))
        new{T,D}(μ_T, α_T, ω_T, mark_dist)
    end
end

HawkesProcess(μ, α, ω) = HawkesProcess(μ, α, ω, NoMarks())

Base.ndims(hp::MultivariateHawkesProcess) = length(hp.μ)

function ground_intensity(hp::HawkesProcess, h::History, t)
    activation = sum(exp.(hp.ω .* (@view h.times[1:(searchsortedfirst(h.times, t) - 1)])))
    return hp.μ + (hp.α * activation / exp(hp.ω * t))
end

function integrated_ground_intensity(hp::HawkesProcess{T}, h::History, tmin, tmax) where {T}
    U = promote_type(T, typeof(tmin), typeof(tmax))
    times = event_times(h, h.tmin, tmax)
    integral = zero(U)
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
           ((hp.α / hp.ω) * sum(1 .- exp.(-hp.ω .* (duration(h) .- h.times)))) + # Integral of each kernel
           sum(log.([densityof(hp.mark_dist, t, h, m) for (t, m) in zip(h.times, h.marks)])) # Lok likelihood of marks
end

function time_change(h::History{R,M}, hp::HawkesProcess) where {R<:Real,M}
    T = float(R)
    n = nb_events(h)
    n == 0 && return History(T[], zero(T), T(hp.μ * duration(h)), event_marks(h))
    times = zeros(T, n)

    # In this step, `times` is the vector A in Ozaki (1979)
    #     A[1] = 0, A[i] = exp(-ω(tᵢ - tᵢ₋₁)) (1 + A[i - 1])
    for i in 2:n
        times[i] = exp(-hp.ω * (h.times[i] - h.times[i - 1])) * (1 + times[i - 1])
    end
    tmax = exp(-hp.ω * (h.tmax - h.times[end])) * (1 + times[end])

    # Integral of the intensity corresponding to activation functions
    #    Λ(tₙ) - μtₙ  = (α/ω) ((n-1) - A[n])
    for i in eachindex(times)
        times[i] = (hp.α / hp.ω) * ((i - 1) - times[i]) # Add contribution of activation functions
    end
    tmax = (hp.α / hp.ω) * (n - tmax) # Add contribution of activation functions

    # Add integral of base intensity
    times .+= T.(hp.μ .* (h.times .- h.tmin))
    tmax += T(hp.μ * duration(h))

    return History(; times=times, marks=h.marks, tmin=zero(T), tmax=tmax, check_args=false) # A time re-scaled process starts at t=0
end
