rng = Random.seed!(63)

ppuniv = PoissonProcess()
pp = PoissonProcess(rand(rng, 10))
bpp = BoundedPointProcess(PoissonProcess(rand(rng, 10)), 0.0, 1000.0)
pp0 = PoissonProcess(zeros(10))
bpp0 = BoundedPointProcess(pp0, 0.0, 1000.0)

h1 = simulate(rng, pp, 0.0, 1000.0)
h2 = simulate_ogata(rng, pp, 0.0, 1000.0)
h2bis = simulate(rng, bpp)
h3 = simulate(rng, pp0, 0.0, 1000.0)
h4 = simulate_ogata(rng, pp0, 0.0, 1000.0)
h4bis = simulate(rng, bpp0)

pp_est1 = fit(MultivariatePoissonProcess{Float32}, [h1, h1])
pp_est1_bis = fit(MultivariatePoissonProcess{Float32}, h1)
pp_est2 = fit(MultivariatePoissonProcess{Float32}, [h2, h2])

prior = MultivariatePoissonProcessPrior(ones(10), 0.0)
pp_est3 = fit_map(MultivariatePoissonProcess{Float32}, prior, [h1, h2])

λ_error1 = mean(abs, pp_est1.λ - pp.λ)
λ_error2 = mean(abs, pp_est2.λ - pp.λ)
λ_error3 = mean(abs, pp_est3.λ - pp.λ)

l = logdensityof(pp, h1)
l_est = logdensityof(pp_est1, h1)

f1(λ) = logdensityof(PoissonProcess(λ), h1)
gf = ForwardDiff.gradient(f1, 3 * ones(10))
# gz = Zygote.gradient(f1, 3 * ones(10))[1]

@test issorted(event_times(h1))
@test issorted(event_times(h2))
@test issorted(event_times(h2bis))
@test !has_events(h3)
@test !has_events(h4)
@test !has_events(h4bis)
@test DensityKind(pp) == HasDensity()
@test length(pp) == 10
@test length(pp0) == 10
@test λ_error1 < 0.1
@test λ_error2 < 0.1
@test λ_error3 < 0.1
@test l_est > l
# @test all(gf .≈ gz)
@test all(gf .< 0)
# @test all(gz .< 0)
@test string(pp0) ==
    "MultivariatePoissonProcess([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])"
@test intensity_vector(pp0) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
@test string(ppuniv) == "UnivariatePoissonProcess(1.0)"
@test_throws DomainError(
    "λ = [-1.0, 1.0]",
    "PoissonProcess: the condition λ ≥ 0 is not satisfied for all dimensions.",
) PoissonProcess([-1.0, 1.0])
