"""
    HawkesProcess <: AbstractPointProcess

Common interface for all subtypes of `HawkesProcess`.
"""
abstract type HawkesProcess{R<:Real,D} <: AbstractPointProcess end

#=
    process_mark(distribution, mark)

For all algorithms, an unmarked process and a process
whose marks are all equal to 1 are equivalent.
This unifies the calculation of intensities and avoids
code repetition.
=#
process_mark(::Any, ::Nothing) = 1.0
process_mark(::Any, m::Real) = m
process_mark(::Type{Dirac{Nothing}}, ::Real) = 1.0

#=
    update_A(α, ω, t, ti_1, mi_1, Ai_1, distribution)

Used for calculating the elements in the vector A in Ozaki (1979). There, the
definition of A is
    A[0] = 0
    A[i] = exp(-ω (tᵢ - tᵢ₋₁)) * (1 + A[i - 1])
and λ(tᵢ) = μ + α A[i].
Here we make two modifications.
First, if α is not fixed for all events, that is, we have α₁, ..., αₙ
corresponding to each of the events t₁, ..., tₙ (in the marked case, 
αᵢ = αmᵢ, mᵢ the i-th mark), then we define
    A[0] = 0
    A[i] = exp(-ω (tᵢ - tᵢ₋₁)) * (αᵢ + A[i - 1])
so λ(tᵢ) = μ + A[i].
The second is that we allow A to be updated with an arbitrary time t, because we
can calculate λ(t) as λ(t) = μ + At, with
    At = exp(-ω (t - tₖ)) * (αₖ + A[k])
where tₖ is the last event time of the process before t.
=#
function update_A(α::R, ω::R, t::Real, ti_1::Real, Ai_1)::float(R) where {R<:Real}
    return exp(-ω * (t - ti_1)) * (α + Ai_1)
end
