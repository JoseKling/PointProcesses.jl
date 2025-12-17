# Constructor
@test HawkesProcess(1, 1, 2) isa UnmarkedUnivariateHawkesProcess{Int64}
@test HawkesProcess(1, 1, 2.0) isa UnmarkedUnivariateHawkesProcess{Float64}
@test_throws DomainError HawkesProcess(-1, 1, 2)
@test_warn r"may cause problems" HawkesProcess(1, 1, 1)

# Ground intensity
hp = HawkesProcess(1, 1, 2)
h = History(; times=[1.0, 2.0, 4.0], marks=[3, 2, 1], tmin=0.0, tmax=5.0)
h_big = History(;
    times=BigFloat.([1, 2, 4]), marks=[3.0, 2.0, 1.0], tmin=BigFloat(0), tmax=BigFloat(5)
)

@test ground_intensity(hp, h, 1) ≈ 1
@test ground_intensity(hp, h, 2) ≈ 1 + hp.α * exp(-hp.ω * 1)
# @test ground_intensity(hp, h, [1, 2, 3, 4]) ≈
#     [ground_intensity(hp, h, t) for t in [1, 2, 3, 4]]

# intensity
@test intensity(hp, "a", 1, h) ≈ ground_intensity(hp, h, 1)
@test intensity(hp, 1, 2, h) ≈ ground_intensity(hp, h, 2)
# @test intensity(hp, 1, [1, 2, 3, 4], h) ≈ [intensity(hp, 1, t, h) for t in [1, 2, 3, 4]]

# Integrated ground intensity
h1 = History([1], 0, 3, [2])
h2 = History([1, 2], 0, 3, [2, 1])

@test integrated_ground_intensity(hp, h1, 0, 1) ≈ 1
@test integrated_ground_intensity(hp, h1, 0, 1000) ≈ 1000.5
@test integrated_ground_intensity(hp, h2, 0, 1000) ≈ 1001
# @test integrated_ground_intensity(hp, h, [1, 2, 3, 4]) ≈
#     [integrated_ground_intensity(hp, h, t) for t in [1, 2, 3, 4]]

# Log likelihood (logintensityof)
@test logdensityof(hp, h1) ≈ log(1) - (hp.μ * duration(h1)) - (1 - exp(-4)) / 2
@test logdensityof(hp, h2) ≈
    log(1) + log(1 + exp(-2)) - (hp.μ * duration(h2)) - ((1 - exp(-4)) / 2) -
      ((1 - exp(-2)) / 2)

# time change
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

# Fit
# Random.seed!(1)
params_true = (100.0, 100.0, 200.0)
model_true = HawkesProcess(params_true...)
h_sim = simulate(model_true, 0.0, 50.0)
model_est = fit(UnmarkedUnivariateHawkesProcess{Float64}, h_sim)
params_est = (model_est.μ, model_est.α, model_est.ω)

@test isa(model_est, UnmarkedUnivariateHawkesProcess)
@test all((params_true .* 0.9) .<= params_est .<= (params_true .* 1.1))
@test isa(
    fit(UnmarkedUnivariateHawkesProcess{Float32}, h_sim),
    UnmarkedUnivariateHawkesProcess{Float32},
)

# simulate
h_sim = simulate(hp, 0.0, 10.0)
@test issorted(h_sim.times)
@test isa(h_sim, History{Float64,Nothing})
