using Transducers
struct Batch{T<:AbstractVector}
    field::T
end
"""
	ConcatenatedBatch

Represent a vector of arrays of sizes (a..., b_n) where b_n is the variable dimension of the batch.
You can access view of individual arrays with `get_slice`.
"""
struct ConcatenatedBatch{T<:AbstractArray}
    field::T
    lengths::Vector{Int}
    function ConcatenatedBatch(field::T, lengths::Vector{Int}) where {T<:AbstractArray}
        @assert first(lengths) == 0 "got $lengths"
        @assert issorted(lengths) "got $lengths"
        @assert last(lengths) == size(field)[end] "got $lengths, size is $( size(field))"
        new{T}(field, lengths)
    end
end
Base.length(x::ConcatenatedBatch) = length(x.lengths)
function ConcatenatedBatch((; field)::Batch)
    ConcatenatedBatch(cat(field; dims=ndims(field)), vcat([0], field .|> size .|> last |> cumsum))
end
function stack_ConcatenatedBatch(x::AbstractVector{ConcatenatedBatch{T}})::ConcatenatedBatch{T} where T
	field = mapreduce((a,b) ->cat(a, b; dims=ndims(a)), x) do a 
		a.field
    end
    offsets = vcat([0], getfield.(x, :lengths) .|> last)::Vector{Int} |> cumsum |> DropLast(1)
    lengths = vcat([0], reduce(vcat,
        zip(getfield.(x, :lengths) |> Map(Drop(1)), offsets) |> Map(((lengths, offset),) -> lengths .+ offset)
    ))

    ConcatenatedBatch(field, lengths)
end
get_slice(lengths::Vector{Int}, i::Integer) = (lengths[i]+1):lengths[i+1]
get_slice(lengths::Vector{Int}, i::UnitRange) = (lengths[minimum(i)]+1):lengths[maximum(i)+1]
function get_element((; field, lengths)::ConcatenatedBatch, i::Integer)
	field[fill(:, ndims(field) - 1)..., get_slice(lengths, i)]
end
function get_element((; field, lengths)::ConcatenatedBatch, i::UnitRange)
    idx = lengths[minimum(i):(maximum(i)+1)]
    idx .-= first(idx)
	ConcatenatedBatch(field[fill(:, ndims(field) - 1)..., get_slice(lengths, i)], idx)
end
