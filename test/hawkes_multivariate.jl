# Constructor
@test HawkesProcess(rand(Float32, 3), rand(Float32, 3), rand(Float32, 3) .+ 1) isa
    MultivariateHawkesProcess{Float32}
@test_throws DomainError HawkesProcess(rand(3) .- 1, rand(3, 3), rand(3) .+ 1)
@test_throws DimensionMismatch HawkesProcess(rand(3), rand(2, 2), rand(3) .+ 1)
@test_warn r"may cause problems" HawkesProcess(rand(3), rand(3, 3) .+ 1, rand(3))

hp = HawkesProcess([1, 2, 3], [1 0.1 0.1; 0.2 2 0.2; 0.3 0.3 3], [2, 4, 6])
h = History(; times=[1.0, 2.0, 4.0], marks=[3, 2, 1], tmin=0.0, tmax=5.0)
h_big = History(;
    times=BigFloat.([1, 2, 4]), marks=[3, 2, 1], tmin=BigFloat(0), tmax=BigFloat(5)
)

# Ground intensity
@test ground_intensity(hp, h, 1) ≈ 6
@test ground_intensity(hp, h, 2) ≈ 6 + sum(hp.α[3, :] .* exp.(-hp.ω .* 1))
# @test ground_intensity(hp, h, [1, 2, 3, 4]) ≈
#     [ground_intensity(hp, h, t) for t in [1, 2, 3, 4]]

# intensity
@test intensity(hp, 1, 1, h) ≈ 1
@test intensity(hp, 1, 2, h) ≈ 1 + hp.α[3, 1] * exp(-hp.ω[1] * 1)
@test intensity(hp, 4, 2, h) == 0
# @test intensity(hp, 1, [1, 2, 3, 4], h) ≈ [intensity(hp, 1, t, h) for t in [1, 2, 3, 4]]

h1 = History([1], 0, 3, [2])
h2 = History([1, 2], 0, 3, [2, 1])

# Integrated ground intensity
@test integrated_ground_intensity(hp, h1, 0, 1) ≈ 6
@test integrated_ground_intensity(hp, h1, 0, 1000) ≈ 6000 + sum(hp.α[2, :] ./ hp.ω)
@test integrated_ground_intensity(hp, h2, 0, 1000) ≈
    6000 + sum(hp.α[2, :] ./ hp.ω) + sum(hp.α[1, :] ./ hp.ω)
# @test integrated_ground_intensity(hp, h, [1, 2, 3, 4]) ≈
#     [integrated_ground_intensity(hp, h, t) for t in [1, 2, 3, 4]]

# Log likelihood (logintensityof)
@test logdensityof(hp, h1) ≈
    log(2) - (hp.μ * duration(h1)) - sum((hp.α[2, :] ./ hp.ω) .* (1 .- exp.(-2 .* hp.ω)))
@test logdensityof(hp, h2) ≈
    log(2) + log(1 + hp.α[2, 1] * exp(-hp.ω[1])) - (hp.μ * duration(h2)) -
      sum((hp.α[2, :] ./ hp.ω) .* (1 .- exp.(-2 .* hp.ω))) -
      sum((hp.α[1, :] ./ hp.ω) .* (1 .- exp.(-hp.ω)))

# time change
h_transf = time_change(hp, h1)
integral = maximum(
    ((hp.μ .* probs(hp.mark_dist)) .* duration(h1)) .+
    (hp.α[2, :] ./ hp.ω) .* (1 .- exp.(-2 .* hp.ω)),
)

@test isa(h_transf, History{Float64,Int})
@test h_transf.marks == h1.marks
@test h_transf.tmin ≈ 0
@test h_transf.tmax ≈ integral
@test h_transf.times ≈ [2.0]
@test isa(time_change(hp, h_big), typeof(h_big))

# simulate
h_sim = simulate(hp, 0.0, 10.0)
hp_mult = HawkesProcess([1, 1], [1, 1], [2, 2])
hp_univ = HawkesProcess(1, 1, 2)
n_sims = 10000
mean_mult = sum([nb_events(simulate(hp_mult, 0, 10)) for _ in 1:n_sims]) / (2 * n_sims)
mean_univ = sum([nb_events(simulate(hp_univ, 0, 10)) for _ in 1:n_sims]) / n_sims

@test issorted(h_sim.times)
@test isa(h_sim, History{Float64,Int64})
@test isapprox(mean_mult, mean_univ, atol=1)
