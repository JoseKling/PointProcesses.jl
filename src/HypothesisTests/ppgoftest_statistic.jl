abstract type PPGoFTest <: HypothesisTest end

function pvalue(test::PPGoFTest) end

abstract type Statistic end

function statistic end
