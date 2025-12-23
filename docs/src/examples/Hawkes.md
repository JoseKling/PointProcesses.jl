```@meta
EditURL = "../../examples/Hawkes.jl"
```

# Fitting data to a Hawkes Process Model

This tutorial demonstrates how to fit data to a Hawkes point process model using maximum likelihood estimation (MLE). We will go over the basics of Hawkes processes,
why they are useful, and how to implement them in PointProcesses.jl. We will first begin with a recap of the math of a general Hawkes process.

## A Quick Recap of Hawkes Processes
A Hawkes process is a type of temporal point process that models self-exciting behavior, where the occurrence of an event increases the likelihood of future events occurring in the near future.
The intensity function of a general Hawkes process is given by:
``λ(t|\mathcal H_t) = μ(t) + ∑_{tᵢ < t} \phi(t - tᵢ)``
where μ(t) is the baseline intensity, which accounts for spontaneous events, ``\mathcal H_t`` is the history of events up until time ``t``, and ``\phi(\cdot)`` is the triggering kernel,
which controls how past events influence future events.

One of the most common choices for this kernel is the exponential function: ``\phi(s) = \alpha\exp{-\beta s}``. In this model, α is the strength of self-exciting, and β controls the rate of excitation decay.
Given these, parameters, the intensity function is: ``λ(t|\mathcal H_t) = μ + \sum_{t_i < t} \alpha \exp{-\beta (t - t_i)}``. We assume that μ is constant for simplicity. But, we could generalzie this to be inhomogeneous.

An important statistic of the Hawkes proces, is the branching ratio which is a measure of the expected number of "daughter" events, a "parent" event will create. For the exponential kernel, this is given by the following equation:
``n = \frac{\alpha}{\beta}``. In the event that n < 1, the process is subcritical, meaning that events will eventually die out. If n = 1, the process is critical, and if n > 1, the process is supercritical, meaning that events can lead to an infinite cascade of events.

Let's now apply this theory, and develop more through the process, by fitting a Hawkes process to some data.

````@example Hawkes
using CSV
using DataFrames
using PointProcesses
using Plots
````

