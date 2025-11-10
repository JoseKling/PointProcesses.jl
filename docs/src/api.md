# API reference

```@docs
PointProcesses
```

## Histories

```@docs
History
```

### Analysis

```@docs
event_marks
event_times
min_time
max_time
nb_events
has_events
Base.length
duration
min_mark
max_mark
```

### Modification

```@docs
push!
append!
cat
time_change
split_into_chunks
```

## Point processes

```@docs
AbstractPointProcess
BoundedPointProcess
```

### Intensity

```@docs
intensity
ground_intensity
log_intensity
```

### Marks

```@docs
mark_distribution
```

### Simulation

```@docs
simulate_ogata
simulate
```

### Inference

```@docs
logdensityof
integrated_ground_intensity
ground_intensity_bound
fit
fit_map
```

## Poisson processes

```@docs
AbstractPoissonProcess
```

### Multivariate

```@docs
MultivariatePoissonProcess
MultivariatePoissonProcessPrior
```

### Marked

```@docs
MarkedPoissonProcess
```

## Hawkes Processes

```@docs
HawkesProcess
```

## Goodness-of-fit tests

```@docs
PPTest
pvalue
Statistic
statistic
BootstrapTest
BootstrapTest(S::Type{<:Statistic}, pp::Type{<:AbstractPointProcess}, h::History; n_sims=1000)
NoBootstrapTest
NoBootstrapTest(S::Type{<:Statistic}, pp::AbstractPointProcess, h::History; n_sims=1000)
```

## Index

```@index
Modules = [PointProcesses]
```
