"""
    HawkesProcess <: AbstractPointProcess

Common interface for all subtypes of `HawkesProcess`.
"""
abstract type HawkesProcess <: AbstractPointProcess end

# #=
# If type parameter for `HawkesProcess` was NOT explicitly provided,
# use `Float64` as the standard type
# =#
# function StatsAPI.fit(
#     HP::Type{<:HawkesProcess}, h::History{H,M}; kwargs...
# ) where {H<:Real,M}
#     T = promote_type(Float64, H)
#     return fit(HP{T}, h; kwargs...)
# end
