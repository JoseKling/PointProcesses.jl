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

## Hawkes Process

```@docs
HawkesProcess
```

## Index

```@index
Modules = [PointProcesses]
```
