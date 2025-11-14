rng = Random.seed!(63)

pp = PoissonProcess(1.0, Normal())
pp0 = PoissonProcess(0.0, Normal())
ppvec = PoissonProcess(2.0, MvNormal(Matrix(I, 2, 2)))

h1 = simulate(rng, pp, 0.0, 1000.0)
h2 = simulate_ogata(rng, pp, 0.0, 1000.0)
h3 = simulate(rng, pp0, 0.0, 1000.0)
h4 = simulate_ogata(rng, pp0, 0.0, 1000.0)

pp_est1 = fit(PoissonProcess{Float32,Normal}, [h1, h1])
pp_est2 = fit(PoissonProcess{Float32,Normal}, [h2, h2])

λ_error1 = mean(abs, pp_est1.λ - pp.λ)
λ_error2 = mean(abs, pp_est2.λ - pp.λ)
μ_error1 = mean(abs, pp_est1.mark_dist.μ - pp.mark_dist.μ)
σ_error1 = mean(abs, pp_est1.mark_dist.σ - pp.mark_dist.σ)
μ_error2 = mean(abs, pp_est2.mark_dist.μ - pp.mark_dist.μ)
σ_error2 = mean(abs, pp_est2.mark_dist.σ - pp.mark_dist.σ)

l = logdensityof(pp, h1)
l_est = logdensityof(pp_est1, h1)

f2(λ) = logdensityof(PoissonProcess(λ, Normal()), h1)
gf = ForwardDiff.derivative(f2, 3)
# gz = Zygote.gradient(f2, 3)[1]

@test issorted(event_times(h1))
@test issorted(event_times(h2))
@test !has_events(h3)
@test !has_events(h4)
@test DensityKind(pp) == HasDensity()
@test λ_error1 < 0.1
@test λ_error2 < 0.1
@test μ_error1 < 0.1
@test σ_error2 < 0.1
@test μ_error1 < 0.1
@test σ_error2 < 0.1
@test l_est > l
# @test all(gf .≈ gz)
@test all(gf .< 0)
# @test all(gz .< 0)
@test string(pp) == "PoissonProcess(1.0, Normal{Float64}(μ=0.0, σ=1.0))"
