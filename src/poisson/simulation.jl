#=
Simulate multivariate Poisson process:
- Generate exponential one interarrival time for each dimension
- Select dimension with earliest event time
- Add event to history and update interarrival time of selected dimension
- Repeat until tmax

The mark distribution can be dependent on the history.
=#
function simulate(rng::AbstractRNG, pp::PoissonProcess, tmin::T, tmax::T) where {T<:Real}
    M = eltype(pp.mark_dist[1])
    R = float(T)
    h = History(fill(R[], ndims(pp)), tmin, tmax, fill(M[], ndims(pp)); check_args=false)
    if ndims(pp) == 1 # Separate for univariate because `push!` takes `nothing` instead of an `Int`
        t = rand(rng, Exponential(1 / pp.λ[1]))
        while t < tmax
            push!(h, t, rand(rng, mark_distribution(pp, t, h, 1)), nothing)
            t += rand(rng, Exponential(1 / pp.λ[1]))
        end
    else
        ts = rand.(rng, Exponential.(1 ./ pp.λ))
        d = argmin(ts)
        while ts[d] < tmax
            push!(h, ts[d], rand(rng, mark_distribution(pp, ts[d], h, d)), d)
            ts[d] += rand(rng, Exponential(1 / pp.λ[d]))
            d = argmin(ts)
        end
    end
    return h
end
