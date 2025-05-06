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
intensity_vector
```

### Marks

```@docs
mark_distribution
```

### Simulation

```@docs
simulate_ogata
Base.rand
```

### Inference

```@docs
logdensityof
```

### Learning

```@docs
integrated_ground_intensity
ground_intensity_bound
fit
fit_map
```

## Poisson processes

```@docs
PoissonProcess
```

### Univariate

```@docs
UnivariatePoissonProcess
```

### Multivariate

```@docs
MultivariatePoissonProcess
MultivariatePoissonProcessPrior
```

## Index

```@index
Modules = [PointProcesses]
```
