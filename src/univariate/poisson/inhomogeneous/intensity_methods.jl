#=
Optimized methods for specific intensity function types.

These provide analytical solutions for bounds and integrals where possible.
=#

## PolynomialIntensity optimizations

#=
Analytical integral for polynomial intensity functions with identity link.
For log link, we use numerical integration.
=#
function integrated_intensity(
    f::PolynomialIntensity{R}, lb::T, ub::T, config::IntegrationConfig
) where {R,T}
    U = promote_type(R, T)
    if f.link === :identity
        result = zero(U)
        for (i, coef) in enumerate(f.coefficients)
            power = i
            result += coef * (ub^power - lb^power) / power
        end
        return result
    else
        integrand(t, p) = f(t)
        prob = IntegralProblem(integrand, (lb, ub))
        integral = solve(
            prob,
            config.solver;
            abstol=config.abstol,
            reltol=config.reltol,
            maxiters=config.maxiters,
        )
        return convert(U, integral.u)
    end
end

#=
Upper bound for polynomial intensity over an interval.

For polynomials, we sample densely and add a margin.
=#
function intensity_bound(
    f::PolynomialIntensity{R},
    t::T;
    lookahead::T=one(T),
    n_samples::Int=10 * length(f.coefficients),
) where {R,T}
    U = promote_type(R, T)
    ts = range(t, t + lookahead; length=n_samples)
    max_val = maximum(f(ti) for ti in ts)
    return (U(max_val * 1.1), U(lookahead))
end

function intensity_bound(
    f::PolynomialIntensity{R},
    t::T,
    h::History;
    lookahead_factor::Real=1/100,
    n_samples::Int=10 * length(f.coefficients),
) where {R,T}
    dur = duration(h)             # or max_time(h) - min_time(h)
    lookahead = T(lookahead_factor * dur)
    return intensity_bound(f, t; lookahead=lookahead, n_samples=n_samples)
end

## ExponentialIntensity optimizations

# Analytical integral for exponential intensity: ∫ a*exp(b*t) dt = (a/b)*(exp(b*ub) - exp(b*lb))
function integrated_intensity(
    f::ExponentialIntensity{R}, lb::T, ub::T, config::IntegrationConfig
) where {R,T}
    U = promote_type(R, T)
    if abs(f.b) < 1e-10
        return U(f.a * (ub - lb))
    end
    return U((f.a / f.b) * (exp(f.b * ub) - exp(f.b * lb)))
end

#=
Upper bound for exponential intensity.

If b > 0 (increasing), max is at right endpoint.
If b < 0 (decreasing), max is at left endpoint.
If b ≈ 0 (constant), use constant bound.
=#
function intensity_bound(f::ExponentialIntensity{R}, t::T; lookahead::T=one(T)) where {R,T}
    U = promote_type(R, T)
    margin = U(1.05)
    if f.b > 1e-10
        # Increasing: max at t + lookahead
        B = f(t + lookahead) * margin
    elseif f.b < -1e-10
        # Decreasing: max at t
        B = f(t) * margin
    else
        # Approximately constant
        B = U(f.a) * margin
    end
    return (U(B), U(lookahead))
end

function intensity_bound(
    f::ExponentialIntensity{R}, t::T, h::History; lookahead_factor::Real=1/100
) where {R,T}
    dur = duration(h)
    lookahead = T(lookahead_factor * dur)
    return intensity_bound(f, t; lookahead=lookahead)
end

## SinusoidalIntensity optimizations

#=
Analytical integral for sinusoidal intensity: ∫ (a + b*sin(ω*t + φ)) dt
=#
function integrated_intensity(
    f::SinusoidalIntensity{R}, lb::T, ub::T, config::IntegrationConfig
) where {R,T}
    U = promote_type(R, T)
    linear_part = f.a * (ub - lb)
    if abs(f.ω) < 1e-10
        sin_part = f.b * sin(f.φ) * (ub - lb)
    else
        sin_part = -(f.b / f.ω) * (cos(f.ω * ub + f.φ) - cos(f.ω * lb + f.φ))
    end
    return U(linear_part + sin_part)
end

#=
Upper bound for sinusoidal intensity.

Maximum is a + |b| (when sin = 1 if b > 0, or sin = -1 if b < 0).
=#
function intensity_bound(f::SinusoidalIntensity{R}, t::T, h::History) where {R,T}
    U = promote_type(R, T)
    B = U(f.a + abs(f.b))
    L = typemax(U)  # Bound holds for all time
    return (B, L)
end

## PiecewiseConstantIntensity optimizations

function integrated_intensity(
    f::PiecewiseConstantIntensity{R}, lb::T, ub::T, config::IntegrationConfig
) where {R,T}
    U = promote_type(R, T)
    start_idx = searchsortedlast(f.breakpoints, lb)
    end_idx = searchsortedlast(f.breakpoints, ub)
    start_idx = max(1, min(start_idx, length(f.rates)))
    end_idx = max(1, min(end_idx, length(f.rates)))

    if start_idx == end_idx
        if start_idx > 0 && start_idx <= length(f.rates)
            return U(f.rates[start_idx] * (ub - lb))
        else
            return zero(U)
        end
    end

    integral = zero(U)

    if start_idx > 0 && start_idx <= length(f.rates)
        integral += f.rates[start_idx] * (f.breakpoints[start_idx + 1] - lb)
    end

    for i in (start_idx + 1):(end_idx - 1)
        if i > 0 && i <= length(f.rates)
            integral += f.rates[i] * (f.breakpoints[i + 1] - f.breakpoints[i])
        end
    end

    if end_idx > 0 && end_idx <= length(f.rates)
        integral += f.rates[end_idx] * (ub - f.breakpoints[end_idx])
    end

    return integral
end

function intensity_bound(f::PiecewiseConstantIntensity{R}, t::T, h::History) where {R,T}
    U = promote_type(R, T)
    idx = searchsortedlast(f.breakpoints, t)

    if idx == 0 || idx >= length(f.breakpoints)
        return (zero(U), typemax(U))
    end

    # Return current rate (exact) and time to next breakpoint.
    # This avoids rejections in Ogata's algorithm within constant pieces.
    current_rate = U(f.rates[idx])
    time_to_next_breakpoint = U(f.breakpoints[idx + 1] - t)

    return (current_rate, time_to_next_breakpoint)
end

## LinearCovariateIntensity

function integrated_intensity(
    f::LinearCovariateIntensity, lb::T, ub::T, config::IntegrationConfig
) where {T}
    integrand(t, p) = f(t)
    prob = IntegralProblem(integrand, (lb, ub))  # 1D
    integral = solve(
        prob,
        config.solver;
        abstol=config.abstol,
        reltol=config.reltol,
        maxiters=config.maxiters,
    )
    return integral.u
end

function intensity_bound(
    f::LinearCovariateIntensity{R}, t::T; lookahead::T=one(T), n_samples::Int=100
) where {R,T}
    U = promote_type(R, T)
    ts = range(t, t + lookahead; length=n_samples)
    max_intensity = maximum(f(ti) for ti in ts)
    return (U(max_intensity * 1.1), U(lookahead))
end

function intensity_bound(
    f::LinearCovariateIntensity{R},
    t::T,
    h::History;
    lookahead_factor::Real=1/100,
    n_samples::Int=100,
) where {R,T}
    dur = duration(h)
    lookahead = T(lookahead_factor * dur)
    return intensity_bound(f, t; lookahead=lookahead, n_samples=n_samples)
end

## Generic fallback for custom functions

# Numerical integral for arbitrary callable intensity functions.
function integrated_intensity(f::F, lb::T, ub::T, config::IntegrationConfig) where {F,T}
    integrand(t, p) = f(t)
    prob = IntegralProblem(integrand, (lb, ub))  # 1D
    integral = solve(
        prob,
        config.solver;
        abstol=config.abstol,
        reltol=config.reltol,
        maxiters=config.maxiters,
    )
    return integral.u
end

# Numerical bound for arbitrary callable intensity functions.
function intensity_bound(f::F, t::T; lookahead::T=one(T), n_samples::Int=100) where {F,T}
    ts = range(t, t + lookahead; length=n_samples)
    max_intensity = maximum(f(ti) for ti in ts)
    U = promote_type(typeof(max_intensity), T)
    return (U(max_intensity * 1.1), U(lookahead))
end

function intensity_bound(
    f::F, t::T, h::History; lookahead_factor::Real=1/100, n_samples::Int=100
) where {F,T}
    dur = duration(h)
    lookahead = T(lookahead_factor * dur)
    return intensity_bound(f, t; lookahead=lookahead, n_samples=n_samples)
end
