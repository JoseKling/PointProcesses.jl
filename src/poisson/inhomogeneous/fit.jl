"""
Fitting methods for inhomogeneous Poisson processes with various intensity functions.

For parametric intensity functions (PolynomialIntensity, ExponentialIntensity, etc.),
we use maximum likelihood estimation. The log-likelihood is:

    ℓ = ∑ᵢ log(λ(tᵢ)) - ∫ₐᵇ λ(t) dt

For most parametric forms, we use numerical optimization.
"""

## PolynomialIntensity fitting

"""
    fit(::Type{InhomogeneousPoissonProcess{PolynomialIntensity{R},M}}, h::History, degree::Int) where {R,M}

Fit an inhomogeneous Poisson process with polynomial intensity to a history.

# Arguments
- `pptype`: The type of the process to fit
- `h`: Event history
- `degree`: Degree of the polynomial (0 for constant, 1 for linear, etc.)

The mark distribution is fitted separately from the event times.
"""
function StatsAPI.fit(
    ::Type{InhomogeneousPoissonProcess{PolynomialIntensity{R},M}}, h::History, degree::Int
) where {R,M}
    times = event_times(h)
    marks = event_marks(h)
    tmin = min_time(h)
    tmax = max_time(h)

    n = length(times)
    if n == 0
        # No events: use zero intensity
        coeffs = zeros(R, degree + 1)
        mark_dist = fit(M, marks)
        return InhomogeneousPoissonProcess(PolynomialIntensity(coeffs), mark_dist)
    end

    # Fit mark distribution
    mark_dist = fit(M, marks)

    # Normalize times to [0, 1] for numerical stability
    times_norm = (times .- tmin) ./ (tmax - tmin)

    # Use least squares to estimate polynomial coefficients
    # We approximate λ(t) by fitting to the empirical rate
    # A more sophisticated approach would use MLE with optimization

    # Simple approach: fit polynomial to cumulative count function
    # N(t) ≈ ∫₀ᵗ λ(s) ds
    cumulative_counts = collect(1:n)

    # Design matrix for polynomial regression
    X = zeros(R, n, degree + 1)
    for i in 1:n
        t_power = one(R)
        for d in 0:degree
            X[i, d + 1] = t_power
            t_power *= times_norm[i]
        end
    end

    # Solve least squares: X * coeffs ≈ cumulative_counts
    # Using normal equations: (X'X) * coeffs = X' * cumulative_counts
    coeffs = (X' * X) \ (X' * cumulative_counts)

    # Scale coefficients back to original time scale
    # If λ_norm(t_norm) = c₀ + c₁*t_norm + c₂*t_norm² + ...
    # and t_norm = (t - tmin) / (tmax - tmin)
    # then λ(t) = (1/scale) * λ_norm((t - tmin)/scale) where scale = tmax - tmin
    scale = tmax - tmin

    # Transform coefficients
    # For derivative (cumulative -> density), we differentiate
    # dN/dt ≈ λ(t)
    coeffs_intensity = zero(coeffs)
    for d in 1:(degree + 1)
        if d <= length(coeffs)
            coeffs_intensity[d] = coeffs[d] * d / scale
        end
    end
    # Shift indices (derivative lowers degree by 1)
    coeffs_intensity = coeffs_intensity[2:end]
    if isempty(coeffs_intensity)
        coeffs_intensity = [R(n / scale)]
    end

    # Ensure positive intensity by taking absolute value of first coefficient
    if !isempty(coeffs_intensity) && coeffs_intensity[1] < 0
        coeffs_intensity[1] = abs(coeffs_intensity[1])
    end

    intensity_func = PolynomialIntensity(coeffs_intensity)
    return InhomogeneousPoissonProcess(intensity_func, mark_dist)
end

"""
    fit(::Type{InhomogeneousPoissonProcess{PolynomialIntensity{R},M}}, h::History) where {R,M}

Fit a linear (degree 1) polynomial intensity by default.
"""
function StatsAPI.fit(
    pptype::Type{InhomogeneousPoissonProcess{PolynomialIntensity{R},M}}, h::History
) where {R,M}
    return fit(pptype, h, 1)  # Default to linear intensity
end

## PiecewiseConstantIntensity fitting

"""
    fit(::Type{InhomogeneousPoissonProcess{PiecewiseConstantIntensity{R},M}}, h::History, n_bins::Int) where {R,M}

Fit an inhomogeneous Poisson process with piecewise constant intensity.

# Arguments
- `pptype`: The type of the process to fit
- `h`: Event history
- `n_bins`: Number of constant intervals to use

Each bin gets a rate estimated as (count in bin) / (duration of bin).
"""
function StatsAPI.fit(
    ::Type{InhomogeneousPoissonProcess{PiecewiseConstantIntensity{R},M}},
    h::History,
    n_bins::Int,
) where {R,M}
    times = event_times(h)
    marks = event_marks(h)
    tmin = min_time(h)
    tmax = max_time(h)

    # Fit mark distribution
    mark_dist = fit(M, marks)

    # Create bins
    breakpoints = collect(range(R(tmin), R(tmax); length=n_bins + 1))

    # Count events in each bin
    counts = zeros(Int, n_bins)
    for t in times
        bin_idx = searchsortedlast(breakpoints, t)
        bin_idx = max(1, min(bin_idx, n_bins))
        counts[bin_idx] += 1
    end

    # Compute rates for each bin
    bin_duration = (tmax - tmin) / n_bins
    rates = R.(counts ./ bin_duration)

    intensity_func = PiecewiseConstantIntensity(breakpoints, rates)
    return InhomogeneousPoissonProcess(intensity_func, mark_dist)
end

"""
    fit(::Type{InhomogeneousPoissonProcess{PiecewiseConstantIntensity{R},M}}, h::History) where {R,M}

Fit with 10 bins by default.
"""
function StatsAPI.fit(
    pptype::Type{InhomogeneousPoissonProcess{PiecewiseConstantIntensity{R},M}}, h::History
) where {R,M}
    return fit(pptype, h, 10)
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
