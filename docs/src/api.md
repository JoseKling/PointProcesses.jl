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
Base.ndims(h::History)
event_marks
event_times
event_dims
min_time
max_time
nb_events
Base.length
has_events
Base.isempty
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
AbstractUnivariateProcess
AbstractMultivariateProcess
BoundedPointProcess
IndependentMultivariateProcess
Base.ndims(::AbstractPointProcess)
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
PoissonProcess
```

### Multivariate

```@docs
MultivariatePoissonProcess
MultivariatePoissonProcessPrior
```

## Inhomogeneous Poisson Process

```@docs
InhomogeneousPoissonProcess
```

### Intensity Functions

```@docs
ParametricIntensity
PolynomialIntensity
ExponentialIntensity
SinusoidalIntensity
PiecewiseConstantIntensity
LinearCovariateIntensity
```

### Configuration

```@docs
from_params
IntegrationConfig
```

## Hawkes Process

```@docs
HawkesProcess
```

## Goodness-of-fit tests

```@docs
Statistic
statistic
PointProcessTest
pvalue
```
### Statistic

```@docs
KSDistance
```

### BootstrapTest

```@docs
BootstrapTest
BootstrapTest(S::Type{<:Statistic}, PP::Type{<:AbstractPointProcess}, h::History)
```

### NoBootstrapTest
```@docs
MonteCarloTest
MonteCarloTest(S::Type{<:Statistic}, pp::AbstractPointProcess, h::History)
```

## Index

```@index
Modules = [PointProcesses]
```
