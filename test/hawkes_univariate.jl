# Constructor
@test HawkesProcess(1, 1, 2, Uniform()) isa UnivariateHawkesProcess{Int64,Uniform{Float64}}
@test_throws DomainError HawkesProcess(1, 1, 2, Uniform(-1, 1))
@test_warn r"may cause problems" HawkesProcess(1, 1, 2, Uniform(10, 11))

hp = HawkesProcess(1, 1, 2, Uniform(0, 3))
h = History(; times=[1.0, 2.0, 4.0], marks=[3, 2, 1], tmin=0.0, tmax=5.0)
h_big = History(;
    times=BigFloat.([1, 2, 4]), marks=[3.0, 2.0, 1.0], tmin=BigFloat(0), tmax=BigFloat(5)
)

# Ground intensity
@test ground_intensity(hp, h, 1) ≈ 1
@test ground_intensity(hp, h, 2) ≈ 1 + 3 * hp.α * exp(-hp.ω * 1)
# @test ground_intensity(hp, h, [1, 2, 3, 4]) ≈
#     [ground_intensity(hp, h, t) for t in [1, 2, 3, 4]]

# Intensity
@test intensity(hp, 1, 1, h) ≈ ground_intensity(hp, h, 1) * (1/3)
@test intensity(hp, 1, 2, h) ≈ ground_intensity(hp, h, 2) * (1/3)
@test intensity(hp, 4, 2, h) == 0
# @test intensity(hp, 1, [1, 2, 3, 4], h) ≈ [intensity(hp, 1, t, h) for t in [1, 2, 3, 4]]

h1 = History([1], 0, 3, [2])
h2 = History([1, 2], 0, 3, [2, 1])

# Integrated ground intensity
@test integrated_ground_intensity(hp, h1, 0, 1) ≈ 1
@test integrated_ground_intensity(hp, h1, 0, 1000) ≈ 1001
@test integrated_ground_intensity(hp, h2, 0, 1000) ≈ 1001.5
# @test integrated_ground_intensity(hp, h, [1, 2, 3, 4]) ≈
#     [integrated_ground_intensity(hp, h, t) for t in [1, 2, 3, 4]]

# Log likelihood (logintensityof)
@test logdensityof(hp, h1) ≈ log(1) - (hp.μ * duration(h1)) - (1 - exp(-4))
@test logdensityof(hp, h2) ≈
    log(1) + log(1 + 2 * exp(-2)) - (hp.μ * duration(h2)) - ((1 - exp(-4))) -
      ((1 - exp(-2)) / 2)

# time change
h_transf = time_change(hp, h)
integral =
    (hp.μ * duration(h)) +
    (hp.α / hp.ω) *
    sum([h.marks[i] * (1 - exp(-hp.ω * (h.tmax - h.times[i]))) for i in 1:length(h.times)])

@test isa(h_transf, typeof(h))
@test h_transf.marks == h.marks
@test h_transf.tmin ≈ 0
@test h_transf.tmax ≈ integral
@test h_transf.times ≈
    [1, 2 + (3 / 2) * (1 - exp(-2)), 4 + (3 / 2) * (1 - exp(-6)) + (1 - exp(-4))]
@test isa(time_change(hp, h_big), typeof(h_big))

# Fit
Random.seed!(1)
model_true = HawkesProcess(params_true..., Uniform())
h_sim = simulate(model_true, 0.0, 50.0)
model_est = fit(UnivariateHawkesProcess{Float32,Uniform}, h_sim)
params_est = (model_est.μ, model_est.α, model_est.ω)
a_est = model_est.mark_dist.a
b_est = model_est.mark_dist.b

@test isa(model_est, UnivariateHawkesProcess)
@test params_true[1] * 0.9 <= params_est[1] <= params_true[1] * 1.1
@test params_true[2] * 0.9 <= params_est[2] <= params_true[2] * 1.1
@test params_true[3] * 0.9 <= params_est[3] <= params_true[3] * 1.1
@test a_est < 0.01
@test b_est > 0.99
@test typeof(model_est.μ) == Float32

# simulate
hp_univ = HawkesProcess(1, 1, 2, Uniform(0, 2))
hp_unmk = HawkesProcess(1, 1, 2)
n_sims = 10000
mean_univ = sum([nb_events(simulate(hp_univ, 0, 10)) for _ in 1:n_sims]) / n_sims
mean_unmk = sum([nb_events(simulate(hp_unmk, 0, 10)) for _ in 1:n_sims]) / n_sims
h_sim = simulate(hp, 0.0, 10.0)

@test issorted(h_sim.times)
@test isa(h_sim, History{Float64,Float64})
