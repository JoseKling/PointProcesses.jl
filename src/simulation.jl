#=
    simulate_poisson_times(rng, λ, tmin, tmax)

Simulate the event times of a homogeneous Poisson process with parameter λ on the interval [tmin, tmax).
Internal function to use in all other simulation algorithms.
=#
function simulate_poisson_times(rng::AbstractRNG, λ, tmin, tmax)
    N = rand(rng, Poisson(λ * (tmax - tmin)))
    times = rand(rng, Uniform(tmin, tmax), N)
    sort!(times)
    return times
end

"""
    simulate_ogata(rng, pp, tmin, tmax)

Simulate a temporal point process `pp` on interval `[tmin, tmax)` using Ogata's algorithm.

# Technical Remark
To infer the type of the marks, the implementation assumes that there is method of `mark_distribution` without the argument `h` such that it corresponds to the distribution of marks in case the history is empty.
"""
function simulate_ogata(
    rng::AbstractRNG, pp::AbstractPointProcess, tmin::T, tmax::T
) where {T<:Real}
    M = typeof(rand(mark_distribution(pp, tmin)))
    h = History(; times=T[], marks=M[], tmin=tmin, tmax=tmax)
    t = tmin
    while t < tmax
        B, L = ground_intensity_bound(pp, t + eps(t), h)
        τ = B > 0 ? rand(rng, Exponential(inv(B))) : typemax(inv(B))
        if τ > L
            t = t + L
        elseif τ <= L
            U_max = ground_intensity(pp, t + τ, h) / B
            U = rand(rng, typeof(U_max))
            if U < U_max
                m = rand(rng, mark_distribution(pp, t + τ, h))
                if t + τ < tmax
                    push!(h, t + τ, m)
                end
            end
            t = t + τ
        end
    end
    return h
end

"""
    rand([rng,] pp, tmin, tmax)

Alias for `simulate_ogata`.
"""
function Base.rand(rng::AbstractRNG, pp::AbstractPointProcess, tmin, tmax)
    return simulate_ogata(rng, pp, tmin, tmax)
end

function Base.rand(pp::AbstractPointProcess, args...; kwargs...)
    return rand(default_rng(), pp, args...; kwargs...)
end
