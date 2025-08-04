abstract type PPGoFTest <: HypothesisTest end

function StatsAPI.pvalue(test::PPGoFTest) end

abstract type Statistic end

function statistic end
