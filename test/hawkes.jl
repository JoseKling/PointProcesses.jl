# Constructor
@test HawkesProcess(1, 1, 2) isa HawkesProcess{Int}
@test HawkesProcess(1, 1, 2.0) isa HawkesProcess{Float64}
@test_throws DomainError HawkesProcess(1, 1, 1)
@test_throws DomainError HawkesProcess(-1, 1, 2)

hp = HawkesProcess(1, 1, 2)
h = History(; times=[1.0, 2.0, 4.0], marks=["a", "b", "c"], tmin=0.0, tmax=5.0)
h_big = History(;
    times=BigFloat.([1, 2, 4]), marks=["a", "b", "c"], tmin=BigFloat(0), tmax=BigFloat(5)
)

# Time change
h_transf = time_change(hp, h)
integral =
    (hp.μ * duration(h)) +
    (hp.α / hp.ω) * sum([1 - exp(-hp.ω * (h.tmax - ti)) for ti in h.times])
@test isa(h_transf, typeof(h))
@test h_transf.marks == h.marks
@test h_transf.tmin ≈ 0
@test h_transf.tmax ≈ integral
@test h_transf.times ≈ [1, (2 + (1 - exp(-2)) / 2), 4 + (2 - exp(-2 * 3) - exp(-2 * 2)) / 2]
@test isa(time_change(hp, h_big), typeof(h_big))

# Ground intensity
@test ground_intensity(hp, h, 1) == 1
@test ground_intensity(hp, h, 2) == 1 + hp.α * exp(-hp.ω * 1)

# Integrated ground intensity
@test integrated_ground_intensity(hp, h, h.tmin, h.tmax) ≈ integral
@test integrated_ground_intensity(hp, h, 2, 3) ≈
    hp.μ +
      ((hp.α / hp.ω) * (exp(-hp.ω) - exp(-hp.ω * 2))) +
      ((hp.α / hp.ω) * (1 - exp(-hp.ω)))

# Rand
h_sim = simulate(hp, 0.0, 10.0)
@test issorted(h_sim.times)
@test isa(h_sim, History{Float64,Nothing})
@test isa(simulate(hp, BigFloat(0), BigFloat(10)), History{BigFloat,Nothing})

# Fit
Random.seed!(123)
params_true = (100.0, 100.0, 200.0)
model = HawkesProcess(params_true...)
h_sim = simulate(model, 0.0, 50.0)
model_est = fit(HawkesProcess, h_sim)
params_est = (model_est.μ, model_est.α, model_est.ω)
@test isa(model_est, HawkesProcess)
@test all((params_true .* 0.9) .<= params_est .<= (params_true .* 1.1))
@test isa(fit(HawkesProcess, h_big), HawkesProcess{BigFloat})
@test isa(fit(HawkesProcess{Float32}, h_big), HawkesProcess{Float32})

# logdensityof
@test logdensityof(hp, h) ≈
    sum(log.(hp.μ .+ (hp.α .* [0, exp(-hp.ω), exp(-hp.ω * 2) + exp(-hp.ω * 3)]))) -
      integral
