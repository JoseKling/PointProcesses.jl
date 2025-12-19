using PointProcesses
using Random
using Distributions

# Create test data
rng = Random.seed!(12345)
gaussian_place_field(t) = 20.0 * exp(-((t - 5.0)^2) / (2 * 1.5^2))
pp_true = InhomogeneousPoissonProcess(gaussian_place_field)
h = simulate(rng, pp_true, 0.0, 10.0)

println("Generated $(length(h)) events")
println("Testing polynomial fit with log link...")

try
    pp_poly = fit(
        PolynomialIntensity{Float64},
        h,
        [2.0, 0.0, -0.1];
        link=:log
    )
    println("✓ Fit successful!")
    println("Coefficients: $(pp_poly.coefficients)")
catch e
    println("✗ Fit failed:")
    showerror(stdout, e)
    println()
end
