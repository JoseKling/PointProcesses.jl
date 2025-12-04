function ground_intensity(hp::UnivariateHawkesProcess, h::History, ts)
    return hp.μ .+ (hp.α .* A_Ozaki_ts(h.times, h.marks, hp.ω, ts))
end

function intensity(
    hp::UnivariateHawkesProcess{R1,D}, m::M, ts, h::History{R2,M}
) where {R1<:Real,R2<:Real,M<:Real,D}
    T = float(promote_type(R1, R2, M, eltype(ts))) # For type stability and coherence
    m in support(hp.mark_dist) || return zero(T)
    return ground_intensity(hp, h, ts) .* T(pdf(hp.mark_dist, m)) # `pdf` always returns a `Float64`
end

function integrated_ground_intensity(hp::UnivariateHawkesProcess, h::History, ts)
    A = A_Ozaki_ts(h.times, h.marks, hp.ω, ts)
    s_marks = sum_marks(h, ts)
    return (hp.μ .* ts) .+ sum((hp.α / hp.ω) .* (s_marks .- A))
end

function integrated_ground_intensity(hp::UnivariateHawkesProcess, h::History, a, b)
    return integrated_ground_intensity(hp, h, b) - integrated_ground_intensity(hp, h, a)
end

function DensityInterface.logdensityof(hp::UnivariateHawkesProcess, h::History)
    A = 0.0
    sum_marks = 0.0
    sum_logλ = 0.0
    for i in 2:nb_events(h)
        sum_marks += h.marks[i - 1]
        A = exp(-hp.ω * (h.times[i] - h.times[i - 1])) * (h.marks[i - 1] + A)
        sum_logλ += log(hp.μ + hp.α * A)
    end
    sum_marks += h.marks[end]
    A = exp(-hp.ω * (h.tmax - h.times[end])) * (h.marks[end] + A)
    Λ_T = (hp.μ * duration(h)) + inv.(hp.ω) * (sum_marks - A)
    return sum_logλ - sum(Λ_T)
end

function sum_marks(h::History, t::Real)
    return sum(h.marks[1:(searchsortedfirst(h.times, t) - 1)])
end

function sum_marks(h::History, ts::Vector{<:Real})
    return [sum(h.marks[1:(searchsortedfirst(h.times, t) - 1)]) for t in ts]
end
