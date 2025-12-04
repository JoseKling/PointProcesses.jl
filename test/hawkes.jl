# Constructor
## UnmarkedUnivariateHawkesProcess
@test HawkesProcess(1, 1, 2) isa UnmarkedUnivariateHawkesProcess{Int64}
@test HawkesProcess(1, 1, 2.0) isa UnmarkedUnivariateHawkesProcess{Float64}
@test_throws DomainError HawkesProcess(1, 1, 1)
@test_throws DomainError HawkesProcess(-1, 1, 2)

## UnivariateHawkesProcess
@test HawkesProcess(1, 1, 2, Uniform()) isa UnivariateHawkesProcess{Int64,Uniform{Float64}}
@test_throws DomainError HawkesProcess(1, 1, 2, Uniform(-1, 1))
@test_throws DomainError HawkesProcess(1, 1, 2, Uniform(10, 11))

## MultivariateHawkesProcess
@test HawkesProcess(rand(Float32, 3), rand(Float32, 3), rand(Float32, 3) .+ 1) isa
    MultivariateHawkesProcess{Float32}
@test_throws DomainError HawkesProcess(rand(3) .- 1, rand(3, 3), rand(3) .+ 1)
@test_throws DomainError HawkesProcess(rand(3), rand(3, 3) .+ 1, rand(3))
@test_throws DimensionMismatch HawkesProcess(rand(3), rand(2, 2), rand(3) .+ 1)

# Time change
hp = HawkesProcess(1, 1, 2)
hp2 = HawkesProcess(1, 1, 2, Uniform(0, 3))
hp3 = HawkesProcess([1, 2, 3], [1 0.1 0.1; 0.2 2 0.2; 0.3 0.3 3], [2, 4, 6])

h = History(; times=[1.0, 2.0, 4.0], marks=[3, 2, 1], tmin=0.0, tmax=5.0)
h_big = History(;
    times=BigFloat.([1, 2, 4]), marks=[3.0, 2.0, 1.0], tmin=BigFloat(0), tmax=BigFloat(5)
)

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
## UnmarkedUnivariateHawkesProcess
@test ground_intensity(hp, h, 1) ≈ 1
@test ground_intensity(hp, h, 2) ≈ 1 + hp.α * exp(-hp.ω * 1)
@test ground_intensity(hp, h, [1, 2, 3, 4]) ≈
    [ground_intensity(hp, h, t) for t in [1, 2, 3, 4]]

## UnivariateHawkesProcess
@test ground_intensity(hp2, h, 1) ≈ 1
@test ground_intensity(hp2, h, 2) ≈ 1 + 3 * hp2.α * exp(-hp2.ω * 1)
@test ground_intensity(hp2, h, [1, 2, 3, 4]) ≈
    [ground_intensity(hp2, h, t) for t in [1, 2, 3, 4]]

## MultivariateHawkesProcess
@test ground_intensity(hp3, h, 1) ≈ 6
@test ground_intensity(hp3, h, 2) ≈ 6 + sum(hp3.α[3, :] .* exp.(-hp3.ω .* 1))
@test ground_intensity(hp3, h, [1, 2, 3, 4]) ≈
    [ground_intensity(hp3, h, t) for t in [1, 2, 3, 4]]

# intensity
## UnmarkedUnivariateHawkesProcess
@test intensity(hp, "a", 1, h) ≈ ground_intensity(hp, h, 1)
@test intensity(hp, 1, 2, h) ≈ ground_intensity(hp, h, 2)
@test intensity(hp, 1, [1, 2, 3, 4], h) ≈ [intensity(hp, 1, t, h) for t in [1, 2, 3, 4]]

## UnivariateHawkesProcess
@test intensity(hp2, 1, 1, h) ≈ ground_intensity(hp2, h, 1) * (1/3)
@test intensity(hp2, 1, 2, h) ≈ ground_intensity(hp2, h, 2) * (1/3)
@test intensity(hp2, 4, 2, h) == 0
@test intensity(hp2, 1, [1, 2, 3, 4], h) ≈ [intensity(hp2, 1, t, h) for t in [1, 2, 3, 4]]

## MultivariateHawkesProcess
@test intensity(hp3, 1, 1, h) ≈ 1
@test intensity(hp3, 1, 2, h) ≈ 1 + hp3.α[3, 1] * exp(-hp3.ω[1] * 1)
@test intensity(hp3, 4, 2, h) == 0
@test intensity(hp3, 1, [1, 2, 3, 4], h) ≈ [intensity(hp3, 1, t, h) for t in [1, 2, 3, 4]]

# Integrated ground intensity
h1 = History([1], 0, 3, [2])
h2 = History([1, 2], 0, 3, [2, 1])

## UnmarkedUnivariateHawkesProcess
@test integrated_ground_intensity(hp, h1, 0, 1) ≈ 1
@test integrated_ground_intensity(hp, h1, 0, 1000) ≈ 1000.5
@test integrated_ground_intensity(hp, h2, 0, 1000) ≈ 1001
# @test integrated_ground_intensity(hp, h, [1, 2, 3, 4]) ≈
#     [integrated_ground_intensity(hp, h, t) for t in [1, 2, 3, 4]]

## UnivariateHawkesProcess
@test integrated_ground_intensity(hp2, h1, 0, 1) ≈ 1
@test integrated_ground_intensity(hp2, h1, 0, 1000) ≈ 1001
@test integrated_ground_intensity(hp2, h2, 0, 1000) ≈ 1001.5
# @test integrated_ground_intensity(hp2, h, [1, 2, 3, 4]) ≈
#     [integrated_ground_intensity(hp2, h, t) for t in [1, 2, 3, 4]]

## MultivariateHawkesProcess
@test integrated_ground_intensity(hp3, h1, 0, 1) ≈ 6
@test integrated_ground_intensity(hp3, h1, 0, 1000) ≈ 6000 + sum(hp3.α[2, :] ./ hp3.ω)
@test integrated_ground_intensity(hp3, h2, 0, 1000) ≈
    6000 + sum(hp3.α[2, :] ./ hp3.ω) + sum(hp3.α[1, :] ./ hp3.ω)
# @test integrated_ground_intensity(hp3, h, [1, 2, 3, 4]) ≈
#     [integrated_ground_intensity(hp3, h, t) for t in [1, 2, 3, 4]]

# logdensityof
# TODO: tests
@test logdensityof(hp, h) ≈
    sum(log.(hp.μ .+ (hp.α .* [0, exp(-hp.ω), exp(-hp.ω * 2) + exp(-hp.ω * 3)]))) -
      integral

# simulate
# TODO: tests
h_sim = simulate(hp, 0.0, 10.0)
@test issorted(h_sim.times)
@test isa(h_sim, History{Float64,Nothing})
@test isa(simulate(hp, BigFloat(0), BigFloat(10)), History{BigFloat,Nothing})

# Fit
Random.seed!(1)
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
