function ground_intensity(
    hp::UnivariateHawkesProcess{R,D}, h::History, t::Real
) where {R<:Real,D}
    T = float(R)
    μ, α, ω = T.((hp.μ, hp.α, hp.ω))

    # If t is outside of history, intensity is 0
    (t < h.tmin || t > h.tmax) && return zero(T)

    # If t is before first event, intensity is just the base intensity
    t <= h.times[1] && return μ

    # Calculate the contribution of the activation functions using Ozaki (1979)
    λt = zero(T)
    ind_h = 2
    while ind_h <= nb_events(h) && t > h.times[ind_h]
        mi_1 = T(process_mark(D, h.marks[ind_h - 1]))
        λt = update_A(mi_1 * α, ω, h.times[ind_h], h.times[ind_h - 1], λt)
        ind_h += 1
    end
    mi_1 = T(process_mark(D, h.marks[ind_h - 1]))
    λt = update_A(mi_1 * α, ω, t, h.times[ind_h - 1], λt)

    # Add the base intensity
    λt += hp.μ

    return λt
end

function intensity(hp::UnivariateHawkesProcess{R}, m, t::Real, h::History) where {R<:Real}
    T = float(R)
    m in support(hp.mark_dist) || return zero(T)
    return ground_intensity(hp, h, t) * T(pdf(hp.mark_dist, m)) # `pdf` always returns a `Float64`
end

function intensity(hp::UnmarkedUnivariateHawkesProcess, _, t::Real, h::History)
    return ground_intensity(hp, h, t)
end

function integrated_ground_intensity(
    hp::UnivariateHawkesProcess{R,D}, h::History, t
) where {R<:Real,D}
    T = float(R)
    μ, α, ω = T.((hp.μ, hp.α, hp.ω))

    # If t is outside of history, intensity is 0
    t <= h.tmin && return zero(T)

    # Add the base intensity
    Λt = μ * T(t)

    # If t is before first event, intensity is just the base intensity
    t <= h.times[1] && return Λt

    # Calculate the contribution of the activation functions using Ozaki (1979)
    A = zero(T)
    sum_marks = zero(T)
    ind_h = 2
    while ind_h <= nb_events(h) && t > h.times[ind_h]
        mi_1 = T(process_mark(D, h.marks[ind_h - 1]))
        sum_marks += mi_1
        A = update_A(mi_1 * α, ω, h.times[ind_h], h.times[ind_h - 1], A)
        ind_h += 1
    end
    mi_1 = T(process_mark(D, h.marks[ind_h - 1]))
    sum_marks += mi_1
    A = update_A(mi_1 * α, ω, t, h.times[ind_h - 1], A)

    # Add intgral of activation functions
    Λt += inv(ω) * (sum_marks - A)

    return Λt
end

function integrated_ground_intensity(hp::UnivariateHawkesProcess, h::History, a, b)
    return integrated_ground_intensity(hp, h, b) - integrated_ground_intensity(hp, h, a)
end

function DensityInterface.logdensityof(
    hp::UnivariateHawkesProcess{R,D}, h::History
) where {R<:Real,D}
    T = float(R)
    μ, α, ω = T.((hp.μ, hp.α, hp.ω))

    A = zero(T)
    sum_marks = zero(T)
    sum_logλ = zero(T)
    for i in 2:nb_events(h)
        mi_1 = T(process_mark(D, h.marks[i - 1]))
        sum_marks += mi_1
        A = update_A(mi_1 * α, ω, h.times[i], h.times[i - 1], A)
        sum_logλ += log(μ + α * A)
    end
    mn = T(process_mark(D, h.marks[end]))
    sum_marks += mn
    A = update_A(mn * α, ω, h.tmax, h.times[end], A)
    Λ_T = (μ * duration(h)) + inv.(ω) * (sum_marks - A)
    return sum_logλ - sum(Λ_T)
end
