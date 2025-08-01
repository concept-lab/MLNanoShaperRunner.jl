using Transducers
"""
   Batch 
A wrapper arround a vector to design a variable dimension
"""
struct Batch{T <: AbstractVector}
    field::T
end
function get_max_size(lengths::Vector{T})::T where T
    @assert length(lengths) >= 1
    if length(lengths) == 1
        zero(T)
    else
        maximum(1:(length(lengths)-1)) do i
            lengths[i+1] - lengths[i]
        end
    end
end
"""
	ConcatenatedBatch

Represent a vector of arrays of sizes (a..., b_n) where b_n is the variable dimension of the batch.
You can access view of individual arrays with `get_slice`.
"""
struct ConcatenatedBatch{T <: AbstractArray,G <:Integer,H <: AbstractVector{G}}
    field::T
    max_set_size::G
    lengths::H

    function ConcatenatedBatch(field::T,max_set_size::G, lengths::H) where {T<:AbstractArray,G<:Integer,H <:AbstractVector{G}}
        new{T,G,H}(field,max_set_size,lengths)
    end
    function ConcatenatedBatch(field::T, lengths::H) where {T<:AbstractArray,H<:AbstractVector{<:Integer}}
        max_set_size = get_max_size(lengths)
        new{T,eltype(H),H}(field, max_set_size,lengths)
    end
    function ConcatenatedBatch(field::T, lengths::H) where {T<:AbstractArray,H<:Vector{<:Integer}}
        @assert first(lengths)==0 "got $lengths first value is not zero"
        @assert issorted(lengths) "got $lengths values are not sorted"
        @assert last(lengths)==size(field)[end] "got $lengths, size is $( size(field)) last value of length is not equal to last dim of field"
        max_set_size = get_max_size(lengths)
        new{T,eltype(H),H}(field, max_set_size,lengths)
    end
    end
Base.length(x::ConcatenatedBatch) = length(x.lengths) -1
function ConcatenatedBatch((; field)::Batch)
    dims = vcat([0], field .|> size .|> last |> cumsum)
    res = similar(first(field),size(first(field))[begin:(end - 1)]...,last(dims))
    for i in eachindex(dims)[begin:(end-1)]
        res[fill(:,ndims(res)-1)...,(dims[i] +1):dims[i+1]] .= field[i]
    end
    ConcatenatedBatch(res, dims)
end
Adapt.@adapt_structure ConcatenatedBatch

function stack_ConcatenatedBatch(x::AbstractVector{<:ConcatenatedBatch{T}})::ConcatenatedBatch{T} where {T}
    field = mapreduce((a, b) -> cat(a, b; dims = ndims(a)), x) do a
        a.field
    end
    offsets = vcat([0], getfield.(x, :lengths) .|> last)::Vector{Int} |> cumsum |>
              DropLast(1)
    lengths = vcat([0],
        reduce(vcat,
            zip(getfield.(x, :lengths) |> Map(Drop(1)), offsets) |>
            Map(((lengths, offset),) -> lengths .+ offset)
        ))

    ConcatenatedBatch(field, lengths)
end
get_slice(lengths::Vector{Int}, i::Integer) = (lengths[i] + 1):lengths[i + 1]
function get_slice(lengths::Vector{Int}, i::UnitRange)
    (lengths[minimum(i)] + 1):lengths[maximum(i) + 1]
end
function get_element((; field, lengths)::ConcatenatedBatch, i::Integer)
    field[fill(:, ndims(field) - 1)..., get_slice(lengths, i)]
end
function get_element((;field,lengths)::ConcatenatedBatch,indices::AbstractVector)
    new_slices_lengths = get_slice.(Ref(lengths),indices) .|> length
    new_lengths =vcat([0],cumsum(new_slices_lengths))
    new_field = similar(field)
    for (i,j) in enumerate(indices)
        colons = fill(:, ndims(field) - 1)
        new_field[colons..., get_slice(new_lengths, i)] .= field[colons..., get_slice(lengths, j)] 
    end
    ConcatenatedBatch(new_field,new_lengths)
end
function get_element((; field, lengths)::ConcatenatedBatch, i::UnitRange)
    idx = lengths[minimum(i):(maximum(i) + 1)]
    idx .-= first(idx)
    ConcatenatedBatch(field[fill(:, ndims(field) - 1)..., get_slice(lengths, i)], idx)
end
