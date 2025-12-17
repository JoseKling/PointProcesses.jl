"""
    time_change(hp::MultivariateHawkesProcess, h::History{R}) where {R<:Real}

Time rescaling for multivariate Hawkes processes.

Notice that the duration of the rescaled processes might be different. The output
is a single `History` on the interval from 0 to the length of the longest rescaled
process.

Assume there are M processes, each of them with Nₘ events, m = 1, ..., M. Then the
m-th rescaled process will be defined on the interval [0, Nₘ).
"""
function time_change(hp::MultivariateHawkesProcess, h::History{R,<:Int}) where {R<:Real}
    T = float(R)
    μ = T.(hp.μ .* (probs(hp.mark_dist)))   # Base intensity for each process
    A = zero(μ)                # For calculating the compensator for each process
    sum_marks = zero(μ)
    times_transformed = zero(h.times)   # The rescaled event times
    times_transformed[1] = (h.times[1] - h.tmin) * μ[h.marks[1]] # No activations before first event
    for i in 2:nb_events(h)
        mi_1 = h.marks[i - 1]
        mi = h.marks[i]
        α = hp.α[mi_1, :]
        sum_marks .+= α
        A .= update_A.(α, hp.ω, h.times[i], h.times[i - 1], A)
        times_transformed[i] =
            ((h.times[i] - h.tmin) * μ[mi]) +       # Integral of base rate
            inv(hp.ω[mi]) * (sum_marks[mi] - A[mi]) # Integral of activations
    end
    mn = h.marks[end]
    α = hp.α[mn, :]
    sum_marks .+= α
    A .= update_A.(α, hp.ω, h.tmax, h.times[end], A)
    ΛT = (μ .* duration(h)) .+ inv.(hp.ω) .* (sum_marks .- A) # Length of the intervals of rescaled processes
    return History(;
        times=T.(times_transformed), tmin=zero(T), tmax=T(maximum(ΛT)), marks=event_marks(h)
    )
end
