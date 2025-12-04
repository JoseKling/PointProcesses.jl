function ground_intensity(hp::MultivariateHawkesProcess, h::History, ts)
    return mapreduce(m -> intensity(hp, m, ts, h), +, support(hp.mark_dist))
end

function intensity(
    hp::MultivariateHawkesProcess{R1}, m::Int, ts, h::History{R2}
) where {R1<:Real,R2<:Real}
    T = float(promote_type(R1, R2, eltype(ts)))
    m in support(hp.mark_dist) || return T.(zero(T))
    λt = A_Ozaki_ts(h.times, hp.α[h.marks, m], hp.ω[m], ts)
    return λt .+ (hp.μ * probs(hp.mark_dist)[m])
end

function integrated_ground_intensity(hp::MultivariateHawkesProcess, h::History, ts)
    marks = support(hp.mark_dist)
    A = map(m -> A_Ozaki_ts(h.times, hp.α[h.marks, m], hp.ω[m], ts), marks)
    s_marks = sum_marks(hp, h, marks, ts)
    return (hp.μ .* ts) .+ sum((inv.(hp.ω) .* (s_marks .- A)))
end

function integrated_ground_intensity(hp::MultivariateHawkesProcess, h::History, a, b)
    return integrated_ground_intensity(hp, h, b) - integrated_ground_intensity(hp, h, a)
end

function DensityInterface.logdensityof(hp::MultivariateHawkesProcess, h::History)
    A = zeros(length(hp.μ))
    sum_marks = zeros(length(hp.μ))
    sum_logλ = 0.0
    for i in 2:nb_events(h)
        m = h.marks[i - 1]
        sum_marks .+= hp.α[:, m]
        A .= exp.(-hp.ω .* (h.times[i] - h.times[i - 1])) .* (hp.α[:, m] .+ A)
        sum_logλ += log(hp.μ[m] + A[m])
    end
    m = h.marks[end]
    sum_marks .+= hp.α[:, m]
    A .= exp.(-hp.ω .* (h.tmax - h.times[end])) .* (hp.α[:, m] .+ A)
    Λ_T = (hp.μ .* durantion(h)) + inv.(hp.ω) .* (sum_marks .- A)
    return sum_logλ - sum(Λ_T)
end

function sum_marks(hp::MultivariateHawkesProcess, h::History, marks, t::Real)
    return map(m -> sum(hp.α[h.marks[1:(searchsortedfirst(h.times, t) - 1)], m]), marks)
end

function sum_marks(hp::MultivariateHawkesProcess, h::History, marks, ts::Vector{<:Real})
    return map(
        m -> [sum(hp.α[h.marks[1:(searchsortedfirst(h.times, t) - 1)], m]) for t in ts],
        marks,
    )
end
