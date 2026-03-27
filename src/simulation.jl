#=
    simulate_poisson_times(rng, λ, tmin, tmax)

Simulate the event times of a homogeneous Poisson process with parameter λ on the interval [tmin, tmax).
Internal function to use in all other simulation algorithms.
=#
function simulate_poisson_times(rng::AbstractRNG, λ, tmin, tmax)
    N = rand(rng, Poisson(λ * (tmax - tmin)))
    times = [rand(rng, Uniform(tmin, tmax)) for _ in 1:N] # rand(rng, Uniform(tmin, tmax), N) always outputs a `Float64`
    sort!(times)
    return times
end

"""
    simulate_ogata(rng, pp, tmin, tmax)

Simulate a temporal point process `pp` on interval `[tmin, tmax)` using Ogata's algorithm.

# Technical Remark
To infer the type of the marks, the implementation assumes that there is a method of `mark_distribution` without the argument `h` such that it corresponds to the distribution of marks in case the history is empty.
"""
function simulate_ogata(
    rng::AbstractRNG, pp::AbstractPointProcess, tmin::T, tmax::T
) where {T<:Real}
    M = eltype(rand.(mark_distribution(pp, tmin)))
    N = ndims(pp)
    D = N == 1 ? Nothing : Int
    h = History(T[], tmin, tmax, M[], D[]; check_args=false)
    t = tmin
    while t < tmax
        BLs = ground_intensity_bound(pp, t + eps(t), h)
        τ = [BLs[d][1] > 0 ? rand(rng, Exponential(inv(BLs[d][1])), N) : typemax(inv(BLs[d][1])) for d in 1:N]
        if all(τ) .> L
            t = t + L
        else
            # get the indices of τ such that τ <= L
            ds = findall(τ .<= L)
            passed = false
            for d in ds
                U_max = ground_intensity(pp, t + τ, h, d) / B
                U = rand(rng, typeof(U_max))
                if U < U_max
                    passed = true
                    m = rand(rng, mark_distribution(pp, t + τ, h))
                    if t + τ < tmax
                        push!(h, t + τ, m; check_args=false)
                    end
                end
            end
            t = t + τ
        end
    end
    return h
end

"""
    simulate([rng,] pp, tmin, tmax)

Alias for `simulate_ogata`.
"""
function simulate(rng::AbstractRNG, pp::AbstractPointProcess, tmin, tmax)
    return simulate_ogata(rng, pp, tmin, tmax)
end

function simulate(pp::AbstractPointProcess, args...; kwargs...)
    return simulate(default_rng(), pp, args...; kwargs...)
end
