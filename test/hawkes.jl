# Constructor
@test HawkesProcess(1, 1, 2) isa HawkesProcess{Int}
@test HawkesProcess(1, 1, 2.0) isa HawkesProcess{Float64}
@test_throws DomainError HawkesProcess(1, 1, 1)
@test_throws DomainError HawkesProcess(-1, 1, 2)

hp = HawkesProcess(1, 1, 2)
h = History([1.0, 2.0, 4.0], ["a", "b", "c"], 0.0, 5.0)
h_big = History(BigFloat.([1, 2, 4]), ["a", "b", "c"], BigFloat(0), BigFloat(5))

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
h_sim = rand(hp, 0.0, 10.0)
@test issorted(h_sim.times)
@test isa(h_sim, History{Nothing,Float64})
@test isa(rand(hp, BigFloat(0), BigFloat(10)), History{Nothing,BigFloat})

# Fit
lines = readlines("hawkes_times.txt")[1:2]
h_fit_times = parse.(Float64, split(lines[1], ","))
h_fit = History(h_fit_times, fill(nothing, length(h_fit_times)), 0.0, 1.0)
μ, α, ω = parse.(Float64, split(lines[2], ","))
hp_est = fit(HawkesProcess, h_fit)
@test isa(hp_est, HawkesProcess)
@test isapprox(μ, hp_est.μ, atol=0.01)
@test isapprox(α, hp_est.α, atol=0.01)
@test isapprox(ω, hp_est.ω, atol=0.01)
@test isa(fit(HawkesProcess, h_big), HawkesProcess{BigFloat})
@test isa(fit(HawkesProcess{Float32}, h_big), HawkesProcess{Float32})

# logdensityof
@test logdensityof(hp, h) ≈
    sum(log.(hp.μ .+ (hp.α .* [0, exp(-hp.ω), exp(-hp.ω * 2) + exp(-hp.ω * 3)]))) -
      integral
