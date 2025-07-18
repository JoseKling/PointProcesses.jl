# Constructor
@test HawkesProcess(1, 1, 1) isa HawkesProcess{Int}
@test HawkesProcess(1, 1, 1.0) isa HawkesProcess{Float64}
@test_throws DomainError HawkesProcess(-1, 1, 1)

hp = HawkesProcess(1, 1, 2)
h = History([1.0, 2.0, 4.0], ["a", "b", "c"], 0.0, 5.0)

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
# Ground intensity
@test ground_intensity(hp, h, 1) == 1
@test ground_intensity(hp, h, 2) == 1 + hp.α * exp(-hp.ω * 1)
# Integrated ground intensity
@test integrated_ground_intensity(hp, h, h.tmin, h.tmax) ≈ integral
@test integrated_ground_intensity(hp, h, 2, 3) ≈
    hp.μ +
      ((hp.α / hp.ω) * (exp(-hp.ω) - exp(-hp.ω * 2))) +
      ((hp.α / hp.ω) * (1 - exp(-hp.ω)))
# Simulation
h_sim = rand(hp, 0.0, 10.0)
@test issorted(h_sim.times)
@test isa(h_sim, History{Nothing,Float64})
# Estimation
h_fit_times = parse.(Float64, split(read("hawkes_times.txt", String)[2:(end - 1)], ", "))
h_fit = History(h_fit_times, fill(nothing, length(h_fit_times)), 0.0, 1.0)
hp_est = fit(HawkesProcess, h_fit)
@test isa(hp_est, HawkesProcess)
n_sims = 1000
params = zeros(n_sims, 3)
for i in 1:n_sims
    fit_est = fit(HawkesProcess, h_fit)
    params[i, :] .= [fit_est.μ, fit_est.α, fit_est.ω]
end
median_est = vec(median(params; dims=1))
var_est = vec(var(params; dims=1))
# [112.318, 63.4261, 144.67] is the median over 10.000 estimations on this same data
@test all(isapprox.(median_est, [112.318, 63.4261, 144.67]; rtol=0.01))
# [0.0791, 0.00196, 0.196] is the variance over 10.000 estimations on this same data
@test all(isapprox.(var_est, [0.0791, 0.00196, 0.196]; rtol=0.5))

# logdensityof
@test logdensityof(hp, h) ≈
    sum(log.(hp.μ .+ (hp.α .* [0, exp(-hp.ω), exp(-hp.ω * 2) + exp(-hp.ω * 3)]))) -
      integral
