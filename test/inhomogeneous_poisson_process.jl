using DensityInterface
using Distributions
using ForwardDiff
using Optim
using PointProcesses
using Statistics
using StatsAPI
using Test

rng = Random.seed!(12345)

@testset verbose = true "InhomogeneousPoissonProcess" begin
    @testset "PolynomialIntensity" begin
        # Linear intensity: λ(t) = 1 + 0.5*t
        intensity_linear = PolynomialIntensity([1.0, 0.5])
        pp = InhomogeneousPoissonProcess(intensity_linear, Normal())

        @testset "Intensity function evaluation - identity link" begin
            @test intensity_linear(0.0) ≈ 1.0
            @test intensity_linear(1.0) ≈ 1.5
            @test intensity_linear(2.0) ≈ 2.0
        end

        @testset "Intensity function evaluation - log link" begin
            # Linear log: λ(t) = exp(0.5 + 0.1*t)
            intensity_log = PolynomialIntensity([0.5, 0.1]; link=:log)
            @test intensity_log(0.0) ≈ exp(0.5)
            @test intensity_log(1.0) ≈ exp(0.6)
            @test intensity_log(2.0) ≈ exp(0.7)
            # Verify positivity
            @test all(intensity_log(t) > 0 for t in 0.0:0.1:10.0)
        end

        @testset "Link function validation" begin
            @test_throws ArgumentError PolynomialIntensity([1.0, 0.5]; link=:invalid)
        end

        @testset "Interface methods" begin
            h_empty = History(Float64[], 0.0, 10.0, Float64[])
            @test ground_intensity(pp, 0.0, h_empty) ≈ 1.0
            @test ground_intensity(pp, 2.0, h_empty) ≈ 2.0
            @test mark_distribution(pp, 0.0, h_empty) isa Normal
            @test mark_distribution(pp) isa Normal
        end

        @testset "Simulation" begin
            h1 = simulate(rng, pp, 0.0, 10.0)
            h2 = simulate_ogata(rng, pp, 0.0, 10.0)

            @test issorted(event_times(h1))
            @test issorted(event_times(h2))
            @test nb_events(h1) > 0
            @test nb_events(h2) > 0
            @test min_time(h1) == 0.0
            @test max_time(h1) == 10.0
        end

        @testset "Integrated intensity" begin
            h_empty = History(Float64[], 0.0, 10.0, Float64[])
            # ∫₀¹⁰ (1 + 0.5*t) dt = [t + 0.25*t²]₀¹⁰ = 10 + 25 = 35
            integral = integrated_ground_intensity(pp, h_empty, 0.0, 10.0)
            @test integral ≈ 35.0 rtol = 1e-3
        end

        @testset "Quadratic intensity" begin
            # λ(t) = 1 + 0.5*t + 0.1*t²
            intensity_quad = PolynomialIntensity([1.0, 0.5, 0.1])
            pp_quad = InhomogeneousPoissonProcess(intensity_quad, Categorical([0.3, 0.7]))

            @test intensity_quad(0.0) ≈ 1.0
            @test intensity_quad(1.0) ≈ 1.6
            @test intensity_quad(2.0) ≈ 2.4

            h = simulate(rng, pp_quad, 0.0, 5.0)
            @test issorted(event_times(h))
            @test all(x -> x ∈ [1, 2], event_marks(h))
        end
    end

    @testset "ExponentialIntensity" begin
        # λ(t) = 2*exp(0.1*t)
        intensity_exp = ExponentialIntensity(2.0, 0.1)
        pp = InhomogeneousPoissonProcess(intensity_exp, Uniform())

        @testset "Intensity function evaluation" begin
            @test intensity_exp(0.0) ≈ 2.0
            @test intensity_exp(1.0) ≈ 2.0 * exp(0.1)
            # Verify positivity for all t
            @test all(intensity_exp(t) > 0 for t in -10.0:0.1:10.0)
        end

        @testset "Positivity enforcement" begin
            @test_throws ArgumentError ExponentialIntensity(-1.0, 0.1)
            @test_throws ArgumentError ExponentialIntensity(0.0, 0.1)
            # Negative b is fine (decreasing intensity)
            @test ExponentialIntensity(2.0, -0.1) isa ExponentialIntensity
        end

        @testset "Simulation" begin
            h = simulate(rng, pp, 0.0, 10.0)
            @test issorted(event_times(h))
            @test nb_events(h) > 0
        end

        @testset "Integrated intensity" begin
            h_empty = History(Float64[], 0.0, 10.0, Float64[])
            # ∫ 2*exp(0.1*t) dt = 20*(exp(0.1*t))
            # From 0 to 10: 20*(exp(1) - exp(0)) = 20*(e - 1)
            expected = 20.0 * (exp(1.0) - 1.0)
            integral = integrated_ground_intensity(pp, h_empty, 0.0, 10.0)
            @test integral ≈ expected rtol = 1e-3
        end
    end

    @testset "SinusoidalIntensity" begin
        # λ(t) = 5 + 2*sin(2π*t)
        intensity_sin = SinusoidalIntensity(5.0, 2.0, 2π, 0.0)
        pp = InhomogeneousPoissonProcess(intensity_sin, Normal())

        @testset "Intensity function evaluation" begin
            @test intensity_sin(0.0) ≈ 5.0
            @test intensity_sin(0.25) ≈ 7.0 rtol = 1e-6
            @test intensity_sin(0.5) ≈ 5.0 rtol = 1e-6
            @test intensity_sin(0.75) ≈ 3.0 rtol = 1e-6
            @test intensity_sin(1.0) ≈ 5.0 rtol = 1e-6
            # Verify positivity
            @test all(intensity_sin(t) >= 0 for t in 0.0:0.01:10.0)
        end

        @testset "Positivity enforcement" begin
            # Valid: a >= |b|
            @test SinusoidalIntensity(5.0, 3.0, 2π) isa SinusoidalIntensity
            @test SinusoidalIntensity(5.0, -3.0, 2π) isa SinusoidalIntensity
            @test SinusoidalIntensity(3.0, 3.0, 2π) isa SinusoidalIntensity

            # Invalid: a < |b|
            @test_throws ArgumentError SinusoidalIntensity(2.0, 3.0, 2π)
            @test_throws ArgumentError SinusoidalIntensity(2.0, -3.0, 2π)
        end

        @testset "Simulation" begin
            h = simulate(rng, pp, 0.0, 10.0)
            @test issorted(event_times(h))
            @test nb_events(h) > 0
        end

        @testset "Integrated intensity" begin
            h_empty = History(Float64[], 0.0, 1.0, Float64[])
            # Over one period, sin integrates to 0, so integral = 5*1 = 5
            integral = integrated_ground_intensity(pp, h_empty, 0.0, 1.0)
            @test integral ≈ 5.0 rtol = 1e-3
        end

        @testset "Intensity bounds" begin
            h_empty = History(Float64[], 0.0, 10.0, Float64[])
            B, L = ground_intensity_bound(pp, 0.0, h_empty)
            @test B >= 7.0  # max is a + |b| = 5 + 2 = 7
            @test L == typemax(Float64)  # Bound holds for all time
        end
    end

    @testset "PiecewiseConstantIntensity" begin
        breakpoints = [0.0, 1.0, 2.0, 3.0]
        rates = [1.0, 3.0, 2.0]
        intensity_pw = PiecewiseConstantIntensity(breakpoints, rates)
        pp = InhomogeneousPoissonProcess(intensity_pw, Categorical([0.5, 0.5]))

        @testset "Intensity function evaluation" begin
            @test intensity_pw(0.5) ≈ 1.0
            @test intensity_pw(1.5) ≈ 3.0
            @test intensity_pw(2.5) ≈ 2.0
        end

        @testset "Simulation" begin
            h = simulate(rng, pp, 0.0, 3.0)
            @test issorted(event_times(h))
            @test nb_events(h) > 0
        end

        @testset "Integrated intensity" begin
            h_empty = History(Float64[], 0.0, 3.0, Float64[])
            # ∫₀³ = 1*(1-0) + 3*(2-1) + 2*(3-2) = 1 + 3 + 2 = 6
            integral = integrated_ground_intensity(pp, h_empty, 0.0, 3.0)
            @test integral ≈ 6.0 rtol = 1e-3
        end

        @testset "Constructor validation" begin
            @test_throws ArgumentError PiecewiseConstantIntensity([0.0, 1.0], [1.0, 2.0])
            @test_throws ArgumentError PiecewiseConstantIntensity([1.0, 0.0], [1.0])
            @test_throws ArgumentError PiecewiseConstantIntensity([0.0, 1.0], [-1.0])
        end
    end

    @testset "LinearCovariateIntensity" begin
        # λ(t) = exp(1.0 + 0.5*t + 0.3*sin(2π*t))
        cov1 = t -> t
        cov2 = t -> sin(2π * t)
        intensity_cov = LinearCovariateIntensity(1.0, [0.5, 0.3], [cov1, cov2])
        pp = InhomogeneousPoissonProcess(intensity_cov, Normal())

        @testset "Intensity function evaluation" begin
            # η(t) = 1.0 + 0.5*t + 0.3*sin(2π*t)
            # λ(t) = exp(η(t))
            @test intensity_cov(0.0) ≈ exp(1.0) rtol = 1e-6

            @test intensity_cov(1.0) ≈ exp(1.0 + 0.5*1.0) rtol = 1e-6

            @test intensity_cov(0.25) ≈ exp(1.0 + 0.5*0.25 + 0.3*1.0) rtol = 1e-6
        end

        @testset "Simulation" begin
            h = simulate(rng, pp, 0.0, 5.0)
            @test issorted(event_times(h))
            @test nb_events(h) > 0
        end

        @testset "Constructor validation" begin
            @test_throws ArgumentError LinearCovariateIntensity(1.0, [0.5], [cov1, cov2])
        end
    end

    @testset "Custom intensity function" begin
        # User-defined function
        custom_func = t -> 1.0 + 0.5 * sin(t)^2
        pp = InhomogeneousPoissonProcess(custom_func, Uniform())

        @testset "Simulation" begin
            h = simulate(rng, pp, 0.0, 10.0)
            @test issorted(event_times(h))
            @test nb_events(h) > 0
        end

        @testset "Interface methods" begin
            h_empty = History(Float64[], 0.0, 10.0, Float64[])
            @test ground_intensity(pp, 0.0, h_empty) ≈ 1.0
            @test ground_intensity(pp, π / 2, h_empty) ≈ 1.5
        end
    end

    @testset "Fitting - PiecewiseConstant" begin
        # Generate data from known piecewise constant process
        breakpoints_true = [0.0, 5.0, 10.0]
        rates_true = [2.0, 5.0]
        intensity_true = PiecewiseConstantIntensity(breakpoints_true, rates_true)
        pp_true = InhomogeneousPoissonProcess(intensity_true, Normal())

        h = simulate(rng, pp_true, 0.0, 10.0)

        # Fit with 2 bins
        pp_est = fit(
            InhomogeneousPoissonProcess{
                PiecewiseConstantIntensity{Float64},Normal,IntegrationConfig
            },
            h,
            2,
        )

        @test pp_est isa InhomogeneousPoissonProcess
        @test pp_est.intensity_function isa PiecewiseConstantIntensity
        @test length(pp_est.intensity_function.rates) == 2

        # Rates should be roughly similar to true rates
        est_rates = pp_est.intensity_function.rates
        @test 0.5 * rates_true[1] <= est_rates[1] <= 2.0 * rates_true[1]
        @test 0.5 * rates_true[2] <= est_rates[2] <= 2.0 * rates_true[2]
    end

    @testset "Fitting - Polynomial (MLE)" begin
        # Generate data from linear process with log link
        intensity_true = PolynomialIntensity([0.5, 0.1]; link=:log)
        pp_true = InhomogeneousPoissonProcess(intensity_true, Categorical([0.4, 0.6]))

        h = simulate(rng, pp_true, 0.0, 50.0)

        # Fit linear intensity with MLE
        # Initial params for degree 1 polynomial: [a0, a1]
        pp_est = fit(
            InhomogeneousPoissonProcess{
                PolynomialIntensity{Float64},Categorical,IntegrationConfig
            },
            h,
            [0.5, 0.1];
            link=:log,
        )

        @test pp_est isa InhomogeneousPoissonProcess
        @test pp_est.intensity_function isa PolynomialIntensity
        @test pp_est.mark_dist isa Categorical
        @test pp_est.intensity_function.link === :log

        # Check mark distribution is reasonable
        @test length(pp_est.mark_dist.p) == 2

        # Parameters should be in the right ballpark
        @test abs(pp_est.intensity_function.coefficients[1] - 0.5) < 0.5
        @test abs(pp_est.intensity_function.coefficients[2] - 0.1) < 0.2
    end

    @testset "Fitting - Exponential (MLE)" begin
        # Generate data from exponential process
        intensity_true = ExponentialIntensity(2.0, 0.05)
        pp_true = InhomogeneousPoissonProcess(intensity_true, Normal())

        h = simulate(rng, pp_true, 0.0, 20.0)

        # Fit exponential intensity with MLE
        # Initial params: [log(a), b]
        pp_est = fit(
            InhomogeneousPoissonProcess{
                ExponentialIntensity{Float64},Normal,IntegrationConfig
            },
            h,
            [log(2.0), 0.05],
        )

        @test pp_est isa InhomogeneousPoissonProcess
        @test pp_est.intensity_function isa ExponentialIntensity
        @test pp_est.mark_dist isa Normal

        # Parameters should be in the right ballpark
        # Allow wide tolerance since this is stochastic data
        @test 0.5 * intensity_true.a <=
            pp_est.intensity_function.a <=
            3.0 * intensity_true.a
        @test abs(pp_est.intensity_function.b - intensity_true.b) < 0.1
    end

    @testset "Fitting - Sinusoidal (MLE)" begin
        # Generate data from sinusoidal process
        intensity_true = SinusoidalIntensity(5.0, 2.0, 2π, 0.0)
        pp_true = InhomogeneousPoissonProcess(intensity_true, Uniform())

        h = simulate(rng, pp_true, 0.0, 10.0)

        # Fit sinusoidal intensity with MLE
        # Initial params: [log(a), b_unconstrained, φ] where a = exp(p1), b = tanh(p2)*a
        pp_est = fit(
            InhomogeneousPoissonProcess{
                SinusoidalIntensity{Float64},Uniform,IntegrationConfig
            },
            h,
            [log(5.0), 0.5, 0.0];
            ω=2π,
        )

        @test pp_est isa InhomogeneousPoissonProcess
        @test pp_est.intensity_function isa SinusoidalIntensity
        @test pp_est.mark_dist isa Uniform
        @test pp_est.intensity_function.ω ≈ 2π

        # Check constraint a >= |b| is satisfied
        @test pp_est.intensity_function.a >= abs(pp_est.intensity_function.b)

        # Parameters should be in a reasonable range
        @test 2.0 <= pp_est.intensity_function.a <= 10.0
        @test abs(pp_est.intensity_function.b) <= pp_est.intensity_function.a
    end

    @testset "Log-density and gradients" begin
        intensity_linear = PolynomialIntensity([1.0, 0.5])
        pp = InhomogeneousPoissonProcess(intensity_linear, Normal())

        h = simulate(rng, pp, 0.0, 10.0)
        l = logdensityof(pp, h)

        @test l isa Real
        @test isfinite(l)

        # Test ForwardDiff compatibility
        f(a) = begin
            pp_test = InhomogeneousPoissonProcess(PolynomialIntensity([a, 0.5]), Normal())
            return logdensityof(pp_test, h)
        end

        g = ForwardDiff.derivative(f, 1.0)
        @test isfinite(g)
    end

    @testset "DensityInterface" begin
        intensity_linear = PolynomialIntensity([1.0, 0.5])
        pp = InhomogeneousPoissonProcess(intensity_linear, Normal())

        @test DensityKind(pp) == HasDensity()
    end

    @testset "Display methods" begin
        intensity_linear = PolynomialIntensity([1.0, 0.5])
        intensity_log = PolynomialIntensity([1.0, 0.5]; link=:log)
        pp = InhomogeneousPoissonProcess(intensity_linear, Normal())

        @test string(intensity_linear) == "PolynomialIntensity([1.0, 0.5])"
        @test string(intensity_log) == "PolynomialIntensity([1.0, 0.5], link=:log)"
        @test occursin("InhomogeneousPoissonProcess", string(pp))
    end
end
