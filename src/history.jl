"""
    History{T<:Real, M}

Linear event histories with temporal locations of type `T` and marks of type `M`.

# Fields

- `times::Vector{T}`: sorted vector of event times
- `tmin::T`: start time
- `tmax::T`: end time
- `marks::Vector{M}`: associated vector of event marks
- `dims::Vector{D}`: associated vector of event dimensions
- `N::Int`: number of dimensions
"""
struct History{T<:Real,M,D}
    times::Vector{T}
    tmin::T
    tmax::T
    marks::Vector{M}
    dims::Vector{D}
    N::Int

    function History(
        times::AbstractVector{R1},
        tmin::R2,
        tmax::R3,
        marks::AbstractVector{M}=fill(nothing, length(times)),
        dims::AbstractVector{D}=fill(nothing, length(times)),
        N::Int=length(unique(dims));
        check_args=true,
    ) where {R1<:Real,R2<:Real,R3<:Real,M,D}
        if check_args
            if tmin >= tmax
                throw(
                    DomainError(
                        (tmin, tmax),
                        "End of interval must be strictly larger than the start.",
                    ),
                )
            end
            if length(marks) != length(times) || length(dims) != length(times)
                throw(
                    DimensionMismatch(
                        "The fields `times`, `marks` and `dims` must have the same length."
                    ),
                )
            end
            if !isempty(times)
                perm = sortperm(times)
                times .= times[perm]
                marks .= marks[perm]
                dims .= dims[perm]
                if times[1] < tmin || times[end] >= tmax
                    @warn "Events outside of provided interval were discarded."
                    il = searchsortedfirst(times, tmin)
                    ir = searchsortedfirst(times, tmax) - 1
                    times = times[il:ir]
                    marks = marks[il:ir]
                    dims = dims[il:ir]
                end
            end
            if N == 1
                dims .= nothing
            else
                if any(d -> d < 1 || d > N, dims)
                    throw(
                        DomainError(
                            dims,
                            "Event dimensions must be between 1 and N, where N is the number of dimensions of the history.",
                        ),
                    )
                end
            end
        end
        T = promote_type(R1, R2, R3)
        return new{T,M,D}(times, tmin, tmax, marks, dims, N)
    end
end

function History(
    times::Vector{Vector{R1}}, tmin::R2, tmax::R3, marks::Vector{Vector{M}}; check_args=true
) where {R1<:Real,R2<:Real,R3<:Real,M}
    vec_times = vcat(times...)
    vec_marks = vcat(marks...)
    perm = sortperm(vec_times)
    if length(times) == 1
        vec_dims = fill(nothing, length(vec_times))
    else
        dims = [fill(i, length(times[i])) for i in 1:length(times)]
        vec_dims = vcat(dims...)
    end
    return History(
        vec_times[perm],
        tmin,
        tmax,
        vec_marks[perm],
        vec_dims[perm],
        length(times);
        check_args=check_args,
    )
end

function History(
    times::Vector{Vector{R1}}, tmin::R2, tmax::R3; check_args=true
) where {R1<:Real,R2<:Real,R3<:Real}
    nots = [fill(nothing, length(times[i])) for i in 1:length(times)]
    return History(times, tmin, tmax, nots; check_args=check_args)
end

function History(; times, tmin, tmax, marks=nothing, dims=nothing, check_args=true)
    if times isa Vector{<:Real}
        marks === nothing && (marks = fill(nothing, length(times)))
        dims === nothing && (dims = fill(nothing, length(times)))
        return History(times, tmin, tmax, marks, dims; check_args=check_args)
    else
        marks === nothing &&
            (marks = [fill(nothing, length(times[i])) for i in 1:length(times)])
        return History(times, tmin, tmax, marks; check_args=check_args)
    end
end

function Base.show(io::IO, h::History{T,M}) where {T,M}
    return print(
        io, "History{$T,$M} with $(nb_events(h)) events on interval [$(h.tmin), $(h.tmax))"
    )
end

function History(tmin::R1, tmax::R2, N::Int=1) where {R1<:Real, R2<:Real}
    R = promote_type(R1, R2)
    History(R[], tmin, tmax, [], [], N)
end

"""
    event_times(h)

Return the sorted vector of event times for `h`.
"""
event_times(h::History) = h.times

"""
    event_times(h, d)

Return the sorted vector of event times for `h` in dimension `d`.
"""
function event_times(h::History, d::Int)
    h.N == 1 && d == 1 ? h.times : (@view h.times[h.dims .== d])
end

"""
    event_times(h, tmin, tmax)

Return the sorted vector of event times in the half-open interval `[tmin, tmax)` in `h`.
"""
function event_times(h::History, tmin::Real, tmax::Real)
    i_min = searchsortedfirst(h.times, tmin)
    i_max = searchsortedfirst(h.times, tmax)
    return @view h.times[i_min:(i_max - 1)]
end

"""
    event_times(h, tmin, tmax, d)

Return the sorted vector of event times in the half-open interval `[tmin, tmax)` in dimension `d` of `h`.
"""
function event_times(h::History, tmin::Real, tmax::Real, d::Int)
    times = event_times(h, d)::AbstractVector
    i_min = searchsortedfirst(times, tmin)
    i_max = searchsortedfirst(times, tmax)
    return @view times[i_min:(i_max - 1)]
end

"""
    event_marks(h)

Return the vector of event marks for `h`, sorted according to their event times.
"""
event_marks(h::History) = h.marks

"""
    event_marks(h, d)

Return the vector of event marks in dimension `d` of `h`, sorted according to their event times.
"""
function event_marks(h::History, d::Int)
    h.N == 1 && d == 1 ? h.marks : (@view h.marks[h.dims .== d])
end

"""
    event_marks(h, tmin, tmax)

Return the sorted vector of marks of events in the half-open interval `[tmin, tmax)` in `h`.
"""
function event_marks(h::History, tmin::Real, tmax::Real)
    i_min = searchsortedfirst(h.times, tmin)
    i_max = searchsortedfirst(h.times, tmax)
    return @view h.marks[i_min:(i_max - 1)]
end

"""
    event_marks(h, tmin, tmax, d)

Return the sorted vector of marks of events in the half-open interval `[tmin, tmax)` in dimension `d` of `h`.
"""
function event_marks(h::History, tmin::Real, tmax::Real, d::Int)
    marks = event_marks(h, d)
    times = event_times(h, d)::AbstractVector
    i_min = searchsortedfirst(times, tmin)
    i_max = searchsortedfirst(times, tmax)
    return @view marks[i_min:(i_max - 1)]
end

"""
    ndims(h)

Return the number of dimensions of `h`.
"""
Base.ndims(h::History) = h.N

"""
    event_dims(h)

Return the vector of event dimensions for `h`, sorted according to their event times.
"""
event_dims(h::History) = h.dims

"""
    event_dims(h, tmin, tmax)

Return the vector of event dimensions for events between `tmin` and `tmax` in `h`, sorted according to their event times.
"""
function event_dims(h::History, tmin::Real, tmax::Real)
    i_min = searchsortedfirst(h.times, tmin)
    i_max = searchsortedfirst(h.times, tmax)
    return @view h.dims[i_min:(i_max - 1)]
end

"""
    min_time(h::History)

Return the starting time of `h` (not the same as the first event time).
"""
min_time(h::History) = h.tmin

"""
    max_time(h::History)

Return the end time of `h` (not the same as the last event time).
"""
max_time(h::History) = h.tmax

"""
    nb_events(h)

Count events in `h`.
"""
nb_events(h::History) = length(h.times)

"""
    nb_events(h, d)

Count events in dimension `d` of `h`.
"""
nb_events(h::History, d::Int) = h.N == 1 && d == 1 ? length(h.times) : count(==(d), h.dims)

"""
    nb_events(h, tmin, tmax)

Count events in `h` during the interval `[tmin, tmax)`.
"""
function nb_events(h::History, tmin::Real, tmax::Real)
    i_min = searchsortedfirst(event_times(h), tmin)
    i_max = searchsortedfirst(event_times(h), tmax)
    return i_max - i_min
end

"""
    nb_events(h, tmin, tmax, d::Int)

Count events in dimension `d` of `h` during the interval `[tmin, tmax)`.
"""
function nb_events(h::History, tmin::Real, tmax::Real, d::Int)
    times = event_times(h, d)::AbstractVector
    i_min = searchsortedfirst(times, tmin)
    i_max = searchsortedfirst(times, tmax)
    return i_max - i_min
end

"""
    length(h)

Alias for `nb_events(h)`.
"""
Base.length(h::History) = nb_events(h)

"""
    has_events(h)

check the presence of events in `h`.
`args` can be used to specify dimension and/or time interval.
"""
has_events(h::History, args...) = nb_events(h, args...) > 0

"""
    isempty(h, args...)

check the absence of events in `h`.
`args` can be used to specify dimension and/or time interval.
"""
Base.isempty(h::History, args...) = !has_events(h, args...)

"""
    duration(h)

Compute the difference `h.tmax - h.tmin`.
"""
duration(h::History) = max_time(h) - min_time(h)

"""
    push!(h, t, m)

Add event `(t, m)` inside the interval `[h.tmin, h.tmax)` at the end of history `h`.
"""
function Base.push!(h::History, t::Real, m=nothing, d=nothing; check_args=true)
    if check_args
        @assert h.tmin <= t < h.tmax
        @assert (length(h) == 0) || (h.times[end] < t)
        @assert (d === nothing && h.N == 1) || 1 <= d <= h.N
    end
    push!(h.times, t)
    push!(h.marks, m)
    push!(h.dims, d)
    return nothing
end

"""
    append!(h, ts, ms)

Append events `(ts, ms)` inside the interval `[h.tmin, h.tmax)` at the end of history `h`.
"""
function Base.append!(
    h::History,
    ts::Vector{<:Real},
    ms=fill(nothing, length(ts)),
    ds=fill(nothing, length(ts));
    check_args=true,
)
    if isempty(ts)
        return nothing
    end
    if check_args
        perm = sortperm(ts)
        ts .= ts[perm]
        ms .= ms[perm]
        ds .= ds[perm]
        @assert h.tmin <= ts[1] && ts[end] < h.tmax
        @assert isempty(h) || (h.times[end] <= ts[1])
        @assert length(ts) == length(ms) == length(ds)
        @assert (eltype(ds) == Nothing && h.N == 1) || all(1 .<= ds .<= h.N)
    end
    append!(h.times, ts)
    append!(h.marks, ms)
    append!(h.dims, ds)
    return nothing
end

"""
    cat(h1, h2)

If h1 and h2 are consecutive event histories, i.e., the end of
h1 coincides with the beginning of h2, then create a new event
history by concatenating h1 and h2.
"""
function Base.cat(h1::History, h2::History)
    max_time(h1) ≈ min_time(h2) || throw(
        DomainError(
            (h1.tmax, h2.tmin),
            "End of h1's interval must coincide with start of h2's interval",
        ),
    )
    times = [h1.times; h2.times]
    marks = [h1.marks; h2.marks]
    dims = [h1.dims; h2.dims]
    return History(;
        times=times, tmin=h1.tmin, tmax=h2.tmax, marks=marks, dims=dims, check_args=false
    )
end

"""
    time_change(h, Λ)

Apply the time rescaling `t -> Λ(t)` to history `h`.
"""
function time_change(h::History, Λ)
    new_times = Λ.(event_times(h))
    new_marks = copy(event_marks(h))
    new_tmin = Λ(min_time(h))
    new_tmax = Λ(max_time(h))
    return History(; times=new_times, marks=new_marks, tmin=new_tmin, tmax=new_tmax)
end

"""
    split_into_chunks(h, chunk_duration)

Split `h` into a vector of consecutive histories with individual duration `chunk_duration`.
"""
function split_into_chunks(h::History{T,M,D}, chunk_duration) where {T,M,D}
    chunks = History{T,M,D}[]
    limits = collect(min_time(h):chunk_duration:max_time(h))
    if !(limits[end] ≈ max_time(h))
        push!(limits, max_time(h))
    end
    for (a, b) in zip(limits[1:(end - 1)], limits[2:end])
        times = [t for t in event_times(h) if a <= t < b]
        marks = [m for (t, m) in zip(event_times(h), event_marks(h)) if a <= t < b]
        dims = [d for (t, d) in zip(event_times(h), event_dims(h)) if a <= t < b]
        chunk = History(;
            times=times, marks=marks, dims=dims, tmin=a, tmax=b, check_args=false
        )
        push!(chunks, chunk)
    end
    return chunks
end

"""
    max_mark(h; [init])

Return the largest event mark if it is larger than `init`, and `init` otherwise.
"""
max_mark(h::History; init=first(event_marks(h))) = maximum(event_marks(h); init=init)

"""
    min_mark(h; [init])

Return the smallest event mark if it is smaller than `init`, and `init` otherwise.
"""
min_mark(h::History; init=first(event_marks(h))) = minimum(event_marks(h); init=init)
