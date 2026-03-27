struct PoissonProcessStats{R1<:Real,R2<:Real,M,W}
    nb_events::Vector{R1}
    duration::R2
    marks::Vector{Vector{M}}
    weights::Vector{Vector{W}}
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
    total_nb_events = [mapreduce(
        (h, w) -> w * nb_events(h, d), +, histories, weights; init=zero(W)
    ) for d in 1:ndims(histories[1])]
    total_marks = [mapreduce(h -> event_marks(h, d), vcat, histories) for d in 1:ndims(histories[1])]
    total_weights = [reduce(
        vcat, (fill(w, nb_events(h, d)) for (w, h) in zip(weights, histories))
    ) for d in 1:ndims(histories[1])]
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
