struct MultivariatePoissonProcessStats{R1<:Real,R2<:Real}
    nb_events::Vector{R1}
    duration::R2
end

## Compute sufficient stats

function Distributions.suffstats(
    ::Type{MultivariatePoissonProcess{R}},
    histories::AbstractVector{<:History{<:Real,<:Integer}},
    weights::AbstractVector{W},
) where {R,W}
    m_max = maximum(max_mark(h; init=0) for h in histories)
    total_nb_events = zeros(W, m_max)
    total_duration = zero(W)
    for (h, w) in zip(histories, weights)
        total_duration += w * duration(h)
        for m in event_marks(h)
            total_nb_events[m] += w
        end
    end
    return MultivariatePoissonProcessStats(total_nb_events, total_duration)
end

function Distributions.suffstats(
    ::Type{MultivariatePoissonProcess{R}},
    histories::AbstractVector{<:History{<:Real,<:Integer}},
) where {R}
    weights = ones(length(histories))
    return suffstats(MultivariatePoissonProcess{R}, histories, weights)
end

function Distributions.suffstats(
    pptype::Type{MultivariatePoissonProcess{R}}, h::History{<:Real,<:Integer}
) where {R}
    return suffstats(pptype, [h])
end
