struct PoissonProcessStats{R1<:Real,R2<:Real,M,W}
    nb_events::R1
    duration::R2
    marks::Vector{M}
    weights::Vector{W}
end

## Compute sufficient stats

function Distributions.suffstats(
    ::Type{<:PoissonProcess},
    histories::AbstractVector{<:History},
    weights::AbstractVector{W},
) where {W}
    total_duration = mapreduce(
        (h, w) -> w * duration(h), +, histories, weights; init=zero(W)
    )
    total_nb_events = mapreduce(
        (h, w) -> w * length(h), +, histories, weights; init=zero(W)
    )
    total_marks = mapreduce(event_marks, vcat, histories)
    total_weights = reduce(
        vcat, (fill(w, nb_events(h)) for (w, h) in zip(weights, histories))
    )
    return PoissonProcessStats(total_nb_events, total_duration, total_marks, total_weights)
end

function Distributions.suffstats(
    pptype::Type{<:PoissonProcess}, histories::AbstractVector{<:History}
)
    weights = ones(length(histories))
    return suffstats(pptype, histories, weights)
end

function Distributions.suffstats(pptype::Type{<:PoissonProcess}, h::History)
    return suffstats(pptype, [h])
end
