# # Fitting data to a Hawkes Process Model

# This tutorial demonstrates how to fit data to a Hawkes point process model using maximum likelihood estimation (MLE). We will go over the basics of Hawkes processes,
# why they are useful, and how to implement them in PointProcesses.jl. We will first begin with a recap of the math of a general Hawkes process.

# ## A Quick Recap of Hawkes Processes
# A Hawkes process is a type of temporal point process that models self-exciting behavior, where the occurrence of an event increases the likelihood of future events occurring in the near future.
# The intensity function of a general Hawkes process is given by:
# ``λ(t|\mathcal H_t) = μ(t) + ∑_{tᵢ < t} \phi(t - tᵢ)``
# where μ(t) is the baseline intensity, which accounts for spontaneous events, ``\mathcal H_t`` is the history of events up until time ``t``, and ``\phi(\cdot)`` is the triggering kernel, 
# which controls how past events influence future events. 

# One of the most common choices for this kernel is the exponential function: ``\phi(s) = \alpha\exp{-\beta s}``. In this model, α is the strength of self-exciting, and β controls the rate of excitation decay.
# Given these, parameters, the intensity function is: ``λ(t|\mathcal H_t) = μ + \sum_{t_i < t} \alpha \exp{-\beta (t - t_i)}``. We assume that μ is constant for simplicity. But, we could generalzie this to be inhomogeneous.

# An important statistic of the Hawkes proces, is the branching ratio which is a measure of the expected number of "daughter" events, a "parent" event will create. For the exponential kernel, this is given by the following equation:
# ``n = \frac{\alpha}{\beta}``. In the event that n < 1, the process is subcritical, meaning that events will eventually die out. If n = 1, the process is critical, and if n > 1, the process is supercritical, meaning that events can lead to an infinite cascade of events.

# Let's now apply this theory, and develop more through the process, by fitting a Hawkes process to some data.
using Clustering
using CSV
using DataFrames
using Dates
using Distributions
using PointProcesses
using Plots

# First let's open our data. This data records litter box entries taken from three cats over a period of one month. 
root = pkgdir(PointProcesses)
data = CSV.read(joinpath(root, "docs", "examples", "data", "cats.csv"), DataFrame);

# This dataset has three cats in it, so let's first separate out each cat, using their weight to infer their identity
cat_weights = [parse(Float64, split(i)[1]) for i in data.Value]
clusters = kmeans(reshape(cat_weights, 1, :), 3; maxiter=100, display=:none)

data.CatWeight = cat_weights
data.CatID = clusters.assignments;

# Now that we have set this up we need to get the timestamps in a useable order for plotting. We can use the Dates package for this.
fmt = dateformat"m/d I:M p"
dt = DateTime("12/22 3:09 pm", fmt)

function parse_timestamp_min(s::AbstractString; year=2024)
    dt = DateTime(s, dateformat"m/d I:M p")
    DateTime(year, month(dt), day(dt), hour(dt), minute(dt))
end

data.TimestampDT = parse_timestamp_min.(String.(data.Timestamp))

t0 = minimum(data.TimestampDT)
data.t = (data.TimestampDT .- t0) ./ Minute(1)  # Float64 minutes since first event

# Since this dataset only has resolution up to the nearest minute, we need to check if there are any "ties" as the Hawkes process model requires unique event times.
if any(diff(sort(data.t)) .== 0)
    println(
        "Data contains tied event times. Please add small jitter to event times to make them unique.",
    )
else
    println("Data contains no tied event times. Proceed with fitting. Yay!")
end

# Great! We can now move forward. First, let's visualize the data using a simple event plot.
function eventplot(
    event_times::Vector{Float64};
    title="Event Plot",
    xlabel="Time (minutes)",
    ylabel="Events",
)
    scatter(
        event_times,
        ones(length(event_times));
        markershape=:vline,
        markersize=10,
        label="",
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        yticks=false,
    )
end

cat1_times = data.t[data.CatID .== 1]
cat2_times = data.t[data.CatID .== 2]
cat3_times = data.t[data.CatID .== 3]

p1 = eventplot(cat1_times; title="Cat 1 Litter Box Entries")
p2 = eventplot(cat2_times; title="Cat 2 Litter Box Entries")
p3 = eventplot(cat3_times; title="Cat 3 Litter Box Entries")

plot(p1, p2, p3; layout=(3, 1), size=(800, 600))

# From the plot above, we can see that events qualitatively arrive together. This could mean that if one cat uses the litter box, then 
# another is soon to follow. To further understand the data, let's plot the intensity function for the average day. We can use the Inhomogeneous Poisson Process models
# in PointProcesses.jl to fit a piecewise constant intensity function to the data. This will give us an idea of the average litter box usage over the course of a day.
# While this will not give us any information about self-exciting behavior, it will help us understand the daily patterns of litter box usage. 

tod = minute.(data.TimestampDT) .+ 60 .* hour.(data.TimestampDT) # time-of-day in minutes since midnight
day = Date.(data.TimestampDT)
n_days = length(unique(day))
h_day = History(sort(Float64.(tod)), 0.0, 1440.0) # build a "history" on [0, 1440] minutes
nbins = 96  # 96 bins = 15-minute bins
pp_day = fit(
    InhomogeneousPoissonProcess{PiecewiseConstantIntensity{Float64},Dirac{Nothing}},
    h_day,
    nbins,
)

λ_avg(u) = pp_day.intensity_function(u) / n_days

u = range(0.0, 1440.0; length=2000)
plot(
    u ./ 60,
    λ_avg.(u);
    xlabel="Time of day (hours)",
    ylabel="Empirical intensity (events/min)",
    title="Average-day empirical intensity (piecewise constant)",
    legend=false,
)

# From the plot we can see the intensity function has two peaks, one at around 7-10 AM and the other at 8-10 PM.
# Now, let's fit a Hawkes process to the data. We will for now, ignore the cat identities and fit a single Hawkes process to all the data.
# The goal of this analysis is to understand the self-exciting nature of the litter box entries. I.e., if one cat uses the litter box, 
# does that increase the likelihood of another cat using it soon after? To do this, we can use the implementation in PointProcesses.jl
full_history = History(data.t, 0.0, maximum(data.t) + 1.0)
hawkes_model = fit(HawkesProcess, full_history)

println("Fitted Hawkes Process Parameters:") # hide
println("Base intensity (μ): ", hawkes_model.μ) # hide
println("Excitation parameter (α): ", hawkes_model.α) # hide
println("Decay rate (ω): ", hawkes_model.ω) # hide
println("Branching ratio (n = α/ω): ", hawkes_model.α / hawkes_model.ω) #hide
# We can now evaluate the branching ratio and parameters of the fitted model.
# The branching ratio tells us the expected number of "daughter" events that a "parent" event will create. If this value is significantly greater than 0,
# it suggests that the litter box entries are self-exciting, meaning that one entry increases the likelihood of subsequent entries in the near future.
# If the branching ratio is close to 0, it suggests that the entries are more random and not influenced by previous entries. 
# In this case, we can see that the branching ratio is around 0.42, suggesting a moderate level of self-excitation in the litter box entries.
# Finally, we can visualize the fitted intensity function over time.

ts = sort(data.t)

function λ_hawkes(t::Real)
    hawkes_model.μ +
    sum((hawkes_model.α * exp(-hawkes_model.ω * (t - ti)) for ti in ts if ti < t); init=0.0)
end

u = range(0.0, maximum(ts) + 1.0; length=2000)

plot(
    u,
    λ_hawkes.(u);
    xlabel="Time (minutes)",
    ylabel="Fitted Hawkes intensity (events/min)",
    title="Fitted Hawkes Process Intensity Function",
    legend=false,
)

# From the plot, we can observe how the intensity function varies over time, capturing the self-exciting nature of the litter box entries.
# Like any good statistical analysis, it is important to assess goodness of fit. We will rely on the most useful method in point processes: the time-rescaling theorem.
# Luckily, PointProcesses.jl has inbuilt functionality to assess goodness of fit using this theorem. We will use a Monte Carlo test to compare the empirical KS distance 
# of the rescaled times against simulated data from the fitted Hawkes process. 
test = MonteCarloTest(KSDistance{Exponential}, hawkes_model, full_history; n_sims=1000)

p = pvalue(test) # hide
println("Monte Carlo Test p-value for Hawkes Process fit: ", p) # hide
println(
    "Assuming a significance level of 0.05, we " *
    (p < 0.05 ? "reject" : "fail to reject") *
    "",
) # hide
println("the null hypothesis that the Hawkes process is a good fit to the data.") # hide

# From this analysis, it seems that we can can say that litter-box usage is indeed a self-exciting process, as the fitted Hawkes process provides a good fit to the data.
# It's worth considering that we have ignored the identities of the cats in this analysis. A more thorough analysis could involve fitting a multivariate Hawkes process,
# where each cat is represented as a separate dimension. This would allow us to capture the interactions between the cats more accurately. This will be the subject of a 
# future tutorial.
