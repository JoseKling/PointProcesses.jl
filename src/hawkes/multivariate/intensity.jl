#=
OPTMIZE: If instead of calculating for a single time `t` we want to return
         the intensities for a ORDERED vector fo times, these methods calculating
         be refactored to calculate all the intensities in a single pass.
=#

function ground_intensity(hp::MultivariateHawkesProcess, h::History{<:Real,<:Int}, t::Real)
    return mapreduce(m -> intensity(hp, m, t, h), +, support(hp.mark_dist))
end

#=
OPTMIZE: Calculating `ground_intensity` from `intensity` is cleaner, but
         it is more efficient to calculate directly in one pass.
=#
function intensity(
    hp::MultivariateHawkesProcess{R}, m::Int, t::Real, h::History{<:Real,<:Int}
) where {R<:Real}
    T = float(R)
    m in support(hp.mark_dist) || return zero(T)

    # If t is outside of history, intensity is 0
    (t < h.tmin || t > h.tmax) && return zero(T)

    μ = T(hp.μ * probs(hp.mark_dist)[m])
    α = T.(hp.α[:, m])
    ω = T(hp.ω[m])

    # If t is before first event, intensity is just the base intensity
    t <= h.times[1] && return μ

    # Calculate the contribution of the activation functions using Ozaki (1979)
    λt = zero(T)
    ind_h = 2
    while ind_h <= nb_events(h) && t > h.times[ind_h]
        mi_1 = h.marks[ind_h - 1]
        λt = update_A(α[mi_1], ω, h.times[ind_h], h.times[ind_h - 1], λt)
        ind_h += 1
    end
    mi_1 = h.marks[ind_h - 1]
    λt = update_A(α[mi_1], ω, t, h.times[ind_h - 1], λt)

    # Add the base intensity
    λt += μ

    return λt
end

#=
OPTMIZE: Calculating `integrated`ground`intensity` from `integrated_intensity` is cleaner, but
         it is more efficient to calculate directly in one pass.
=#
function integrated_intensity(
    hp::MultivariateHawkesProcess{R}, m::Int, h::History{<:Real,<:Int}, t::Real
) where {R<:Real}
    T = float(R)
    m in support(hp.mark_dist) || return zero(T)

    # If t occurs before the history starts, then the integral is 0
    t < h.tmin && return zero(T)

    μ = T(hp.μ * probs(hp.mark_dist)[m])
    α = T.(hp.α[:, m])
    ω = T(hp.ω[m])

    Λt = μ * T(t)

    # If t is before first event, intensity is just the base intensity
    t <= h.times[1] && return Λt

    # Calculate the contribution of the activation functions using Ozaki (1979)
    ind_h = 2
    sum_marks = zero(T)
    A = zero(T)
    while ind_h <= nb_events(h) && t > h.times[ind_h]
        mi_1 = h.marks[ind_h - 1]
        sum_marks += α[mi_1]
        A = update_A(α[mi_1], ω, h.times[ind_h], h.times[ind_h - 1], A)
        ind_h += 1
    end
    mi_1 = h.marks[ind_h - 1]
    sum_marks += α[mi_1]
    A = update_A(α[mi_1], ω, t, h.times[ind_h - 1], A)

    # Add contribution of intensity functions
    Λt += inv(ω) * (sum_marks - A)

    return Λt
end

function integrated_intensity(
    hp::MultivariateHawkesProcess, m::Int, h::History{<:Real,<:Int}, a, b
)
    return integrated_intensity(hp, m, h, b) - integrated_intensity(hp, m, h, a)
end

function integrated_ground_intensity(hp, h, t)
    return mapreduce(m -> integrated_intensity(hp, m, h, t), +, support(hp.mark_dist))
end

function integrated_ground_intensity(hp, h, a, b)
    return mapreduce(m -> integrated_intensity(hp, m, h, a, b), +, support(hp.mark_dist))
end

function DensityInterface.logdensityof(
    hp::MultivariateHawkesProcess{R}, h::History{<:Real,<:Int}
) where {R<:Real}
    T = float(R)

    isempty(h.times) && return T(hp.μ * duration(h))

    μ = T.(hp.μ .* probs(hp.mark_dist))
    α = T.(hp.α)
    ω = T.(hp.ω)

    A = zero(μ)
    sum_marks = α[h.marks[1], :]
    sum_logλ = log(μ[h.marks[1]])
    for i in 2:nb_events(h)
        mi_1 = h.marks[i - 1]
        mi = h.marks[i]
        sum_marks .+= α[mi, :]
        A .= update_A.(α[mi_1, :], ω, h.times[i], h.times[i - 1], A)
        sum_logλ += log(μ[mi] + A[mi])
    end
    A .= update_A.(α[h.marks[end], :], ω, h.tmax, h.times[end], A)
    ΛT = (μ .* duration(h)) .+ inv.(ω) .* (sum_marks .- A)
    return sum_logλ - sum(ΛT)
end
