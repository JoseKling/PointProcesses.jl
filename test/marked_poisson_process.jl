rng = Random.seed!(63)

pp = MarkedPoissonProcess(1.0, Categorical([0.1, 0.3, 0.6]))
pp0 = MarkedPoissonProcess(0.0, Categorical([0.1, 0.3, 0.6]))
ppvec = MarkedPoissonProcess(2.0, MvNormal(Matrix(I, 2, 2)))

h1 = simulate(rng, pp, 0.0, 1000.0)
h2 = simulate_ogata(rng, pp, 0.0, 1000.0)
h3 = simulate(rng, pp0, 0.0, 1000.0)
h4 = simulate_ogata(rng, pp0, 0.0, 1000.0)

pp_est1 = fit(MarkedPoissonProcess{Float32,Categorical}, [h1, h1])
pp_est2 = fit(MarkedPoissonProcess{Float32,Categorical}, [h2, h2])

λ_error1 = mean(abs, pp_est1.λ - pp.λ)
λ_error2 = mean(abs, pp_est2.λ - pp.λ)
p_error1 = mean(abs, pp_est1.mark_dist.p - pp.mark_dist.p)
p_error2 = mean(abs, pp_est2.mark_dist.p - pp.mark_dist.p)

l = logdensityof(pp, h1)
l_est = logdensityof(pp_est1, h1)

f2(λ) = logdensityof(MarkedPoissonProcess(λ, Categorical([0.1, 0.3, 0.6])), h1)
gf = ForwardDiff.derivative(f2, 3)
# gz = Zygote.gradient(f2, 3)[1]

@test issorted(event_times(h1))
@test issorted(event_times(h2))
@test !has_events(h3)
@test !has_events(h4)
@test DensityKind(pp) == HasDensity()
@test λ_error1 < 0.1
@test λ_error2 < 0.1
@test p_error1 < 0.1
@test p_error2 < 0.1
@test l_est > l
# @test all(gf .≈ gz)
@test all(gf .< 0)
# @test all(gz .< 0)
string(pp) ==
"MarkedPoissonProcess(1.0, Categorical{Float64, Vector{Float64}}(support=Base.OneTo(3), p=[0.1, 0.3, 0.6]))"
