abstract type Statistic end

function statistic(::Type{<:Statistic}, pp::AbstractPointProcess, h::History) end

abstract type PPGoFTest <: HypothesisTest end

function pvalue(test::PPGoFTest) end
