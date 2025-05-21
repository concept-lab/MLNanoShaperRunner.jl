module MLUtilsExt
using MLUtils, MLNanoShaperRunner
MLUtils.numobs(x::ConcatenatedBatch) = length(x)
MLUtils.getobs(data::ConcatenatedBatch,i) = get_element(data,i)
end
