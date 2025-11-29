"""
Optimized methods for specific intensity function types.

These provide analytical solutions for bounds and integrals where possible.
"""

## PolynomialIntensity optimizations

"""
Analytical integral for polynomial intensity functions with identity link.
For log link, we use numerical integration.
"""
function integrated_intensity(f::PolynomialIntensity, a, b)
    if f.link === :identity
        # Analytical integral for polynomial
        result = zero(promote_type(eltype(f.coefficients), typeof(a), typeof(b)))
        for (i, coef) in enumerate(f.coefficients)
            power = i
            result += coef * (b^power - a^power) / power
        end
        return result
    else
        # Numerical integration for log link
        n_points = max(100, ceil(Int, 100 * (b - a)))
        ts = range(a, b; length=n_points)
        intensities = [f(t) for t in ts]

        integral = zero(eltype(intensities))
        for i in 1:(n_points - 1)
            integral += (intensities[i] + intensities[i + 1]) / 2 * (ts[i + 1] - ts[i])
        end
        return integral
    end
end

"""
Upper bound for polynomial intensity over an interval.

For polynomials, we sample densely and add a margin.
"""
function intensity_bound(f::PolynomialIntensity{R}, t::T) where {R,T}
    lookahead = T(1.0)
    n_samples = 50
    ts = range(t, t + lookahead; length=n_samples)
    max_val = maximum(f(ti) for ti in ts)
    return (max_val * T(1.1), lookahead)
end

## ExponentialIntensity optimizations

"""
Analytical integral for exponential intensity: ∫ a*exp(b*t) dt = (a/b)*(exp(b*b) - exp(b*a))
"""
function integrated_intensity(f::ExponentialIntensity, a, b)
    if abs(f.b) < 1e-10
        # b ≈ 0, treat as constant
        return f.a * (b - a)
    end
    return (f.a / f.b) * (exp(f.b * b) - exp(f.b * a))
end

"""
Upper bound for exponential intensity.

If b > 0 (increasing), max is at right endpoint.
If b < 0 (decreasing), max is at left endpoint.
If b ≈ 0 (constant), use constant bound.
"""
function intensity_bound(f::ExponentialIntensity{R}, t::T) where {R,T}
    lookahead = T(1.0)
    if f.b > 1e-10
        # Increasing: max at t + lookahead
        B = f(t + lookahead) * T(1.05)  # small margin
    elseif f.b < -1e-10
        # Decreasing: max at t
        B = f(t) * T(1.05)
    else
        # Approximately constant
        B = f.a * T(1.05)
    end
    return (B, lookahead)
end

## SinusoidalIntensity optimizations

"""
Analytical integral for sinusoidal intensity: ∫ (a + b*sin(ω*t + φ)) dt
"""
function integrated_intensity(f::SinusoidalIntensity, t_start, t_end)
    linear_part = f.a * (t_end - t_start)
    if abs(f.ω) < 1e-10
        # ω ≈ 0, sin term is approximately constant
        sin_part = f.b * sin(f.φ) * (t_end - t_start)
    else
        sin_part = -(f.b / f.ω) * (cos(f.ω * t_end + f.φ) - cos(f.ω * t_start + f.φ))
    end
    return linear_part + sin_part
end

"""
Upper bound for sinusoidal intensity.

Maximum is a + |b| (when sin = 1 if b > 0, or sin = -1 if b < 0).
"""
function intensity_bound(f::SinusoidalIntensity{R}, t::T) where {R,T}
    B = f.a + abs(f.b)
    L = typemax(T)  # Bound holds for all time
    return (B, L)
end

## PiecewiseConstantIntensity optimizations

"""
Analytical integral for piecewise constant intensity.
"""
function integrated_intensity(f::PiecewiseConstantIntensity, a, b)
    # Find the intervals that overlap with [a, b]
    start_idx = searchsortedlast(f.breakpoints, a)
    end_idx = searchsortedlast(f.breakpoints, b)

    # Clamp to valid range
    start_idx = max(1, min(start_idx, length(f.rates)))
    end_idx = max(1, min(end_idx, length(f.rates)))

    if start_idx == end_idx
        # Entire interval [a, b] is within a single constant region
        if start_idx > 0 && start_idx <= length(f.rates)
            return f.rates[start_idx] * (b - a)
        else
            return zero(eltype(f.rates))
        end
    end

    # Sum over multiple intervals
    integral = zero(eltype(f.rates))

    # First partial interval
    if start_idx > 0 && start_idx <= length(f.rates)
        integral += f.rates[start_idx] * (f.breakpoints[start_idx + 1] - a)
    end

    # Complete intervals in between
    for i in (start_idx + 1):(end_idx - 1)
        if i > 0 && i <= length(f.rates)
            integral += f.rates[i] * (f.breakpoints[i + 1] - f.breakpoints[i])
        end
    end

    # Last partial interval
    if end_idx > 0 && end_idx <= length(f.rates)
        integral += f.rates[end_idx] * (b - f.breakpoints[end_idx])
    end

    return integral
end

"""
Upper bound for piecewise constant intensity.

The bound is simply the maximum rate in the upcoming intervals.
"""
function intensity_bound(f::PiecewiseConstantIntensity{R}, t::T) where {R,T}
    idx = searchsortedlast(f.breakpoints, t)

    if idx == 0 || idx >= length(f.breakpoints)
        # Outside the domain
        return (zero(R), typemax(T))
    end

    # Maximum rate from current position to end
    max_rate = maximum(f.rates[idx:end])

    # Lookahead until the end of the domain
    L = f.breakpoints[end] - t

    return (max_rate, L)
end

## LinearCovariateIntensity

"""
Numerical integral for linear covariate intensity (no general analytical form).
"""
function integrated_intensity(f::LinearCovariateIntensity, a, b)
    # Use trapezoidal rule
    n_points = max(100, ceil(Int, 100 * (b - a)))
    ts = range(a, b; length=n_points)
    intensities = [f(t) for t in ts]

    integral = zero(eltype(intensities))
    for i in 1:(n_points - 1)
        integral += (intensities[i] + intensities[i + 1]) / 2 * (ts[i + 1] - ts[i])
    end

    return integral
end

"""
Numerical bound for linear covariate intensity.
"""
function intensity_bound(f::LinearCovariateIntensity{R}, t::T) where {R,T}
    lookahead = T(1.0)
    n_samples = 100
    ts = range(t, t + lookahead; length=n_samples)
    max_intensity = maximum(f(ti) for ti in ts)
    return (max_intensity * T(1.1), lookahead)
end

## Generic fallback for custom functions

"""
Numerical integral for arbitrary callable intensity functions.
"""
function integrated_intensity(f::F, a, b) where {F}
    n_points = max(100, ceil(Int, 100 * (b - a)))
    ts = range(a, b; length=n_points)
    intensities = [f(t) for t in ts]

    integral = zero(eltype(intensities))
    for i in 1:(n_points - 1)
        integral += (intensities[i] + intensities[i + 1]) / 2 * (ts[i + 1] - ts[i])
    end

    return integral
end

"""
Numerical bound for arbitrary callable intensity functions.
"""
function intensity_bound(f::F, t::T) where {F,T}
    lookahead = T(1.0)
    n_samples = 100
    ts = range(t, t + lookahead; length=n_samples)
    max_intensity = maximum(f(ti) for ti in ts)
    return (max_intensity * T(1.1), lookahead)
end
