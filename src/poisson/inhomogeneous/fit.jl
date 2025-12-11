"""
Fitting methods for inhomogeneous Poisson processes with various intensity functions.

For parametric intensity functions (PolynomialIntensity, ExponentialIntensity, etc.),
we use maximum likelihood estimation. The log-likelihood is:

    ℓ = ∑ᵢ log(λ(tᵢ)) - ∫ₐᵇ λ(t) dt

For most parametric forms, we use numerical optimization with automatic differentiation.
"""

## Generic MLE infrastructure

"""
    negative_loglikelihood_ipp(params, intensity_type, times, tmin, tmax, integration_config; kwargs...)

Compute the negative log-likelihood for an inhomogeneous Poisson process.

This is the objective function to minimize during MLE. It uses the general form:
    -ℓ(θ) = -∑ᵢ log(λ(tᵢ; θ)) + ∫ₐᵇ λ(t; θ) dt

# Arguments
- `h`: the event history
- `f`: A ParametricIntensity intensity function (e.g., ExponentialIntensity{Float64})
- keyword `integration_config`: IntegrationConfig for numerical integration

# Returns
Negative log-likelihood value (to be minimized).
"""
function negative_loglikelihood_ipp(
    h::History, f::ParametricIntensity; integration_config=IntegrationConfig()
)
    # Sum of log intensities at event times (compute first for stability)
    log_sum = mapreduce(t -> log(f(t)), +, h.times)

    # Compute integrated intensity
    Λ = integrated_intensity(f, h.tmin, h.tmax, integration_config)

    # Return negative log-likelihood
    return Λ - log_sum
end

function StatsAPI.fit(
    F::Type{<:ParametricIntensity},
    h::History,
    init_params;
    optimizer=LBFGS(),
    autodiff=:forward,
    integration_config=IntegrationConfig(),
    intensity_kwargs...,
)
    if isempty(h.times)
        return InhomogeneousPoissonProcess(t -> 0.0, Dirac(nothing))
    end

    # Define objective function
    objective(params) = negative_loglikelihood_ipp(
        h,
        from_params(F, params; intensity_kwargs...);
        integration_config=integration_config,
    )

    result = optimize(objective, init_params, optimizer; autodiff=autodiff)

    # Extract optimal parameters
    optimal_params = Optim.minimizer(result)

    return from_params(F, optimal_params; intensity_kwargs...)
end

function StatsAPI.fit(
    ::Type{<:InhomogeneousPoissonProcess{F,D}},
    h::History,
    init_params;
    integration_config=IntegrationConfig(),
    kwargs...,
) where {F<:ParametricIntensity,D}
    # Fit mark distribution independently
    mark_dist = fit(D, h.marks)

    intensity = fit(F, h, init_params; integration_config=integration_config, kwargs...)
    return InhomogeneousPoissonProcess(intensity, mark_dist)
end

## Helper methods for multiple histories

function StatsAPI.fit(
    pptype::Type{<:InhomogeneousPoissonProcess},
    histories::AbstractVector{<:History},
    args...,
)
    # Concatenate all histories
    combined = histories[1]
    for i in 2:length(histories)
        combined = History(
            vcat(event_times(combined), event_times(histories[i])),
            vcat(event_marks(combined), event_marks(histories[i])),
            min(min_time(combined), min_time(histories[i])),
            max(max_time(combined), max_time(histories[i])),
        )
    end
    return fit(pptype, combined, args...)
end
