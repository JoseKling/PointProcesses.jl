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
using PointProcesses
using Plots

# First let's open our data. This data records litter box entries taken from three cats over a period of one month. 
data = CSV.read(joinpath(@__DIR__, "data", "cats.csv"), DataFrame)

# This dataset has three cats in it, so let's first separate out each cat, using their weight to infer their identity
cat_weights = [parse(Float64, split(i)[1]) for i in data.Value]
clusters = kmeans(reshape(cat_weights, 1, :), 3; maxiter=100, display=:none)

data.CatWeight = cat_weights
data.CatID = clusters.assignments

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

# Since this dataset only has resolution up to the nearest minutw, we need to check if there are any "ties" as the Hawkes process model requires unique event times.
if any(diff(sort(data.t)) .== 0)
    println("Data contains tied event times. Please add small jitter to event times to make them unique.")
else
    println("Data contains no tied event times. Proceed with fitting. Yay!")
end

# Great! We can now move forward.