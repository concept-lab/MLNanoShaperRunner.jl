using Lux
using Adapt: @adapt_structure
using ConcreteStructs
using GeometryBasics
using Random
using Adapt
using StructArrays
using Distances
using ChainRulesCore
using Statistics
using Folds
using Transducers
using Static
using StaticTools

function terse end
"""
	ModelInput

input of the model
# Fields
- point::Point3, the position of the input
- atoms::StructVector{Sphere}, the atoms in the neighboord
"""
struct ModelInput{T<:Number}
    point::Point3{T}
    atoms::StructVector{Sphere{T}} #Set
end

#enable running on gpu
Adapt.@adapt_structure Batch
Adapt.@adapt_structure ModelInput
# Adapt.@adapt_structure Partial
@concrete terse struct DeepSet <: Lux.AbstractLuxWrapperLayer{:prepross}
    prepross
end

function (f::DeepSet)(
    (; field,max_set_size, lengths)::ConcatenatedBatch,
    ps::NamedTuple,
    st::NamedTuple
)
    res, st = Lux.apply(f.prepross, field, ps, st)
    ret =  batched_sum(res,max_set_size, lengths)
    # @info res ret lengths
    ret,st
end
function (f::DeepSet)(arg::Batch, ps, st)
    f(ConcatenatedBatch(arg), ps, st)
end
function (f::DeepSet)(set::AbstractArray, ps, st)
    f(Batch([set]), ps, st)
end

@concrete terse struct FixedSizeDeepSet <: Lux.AbstractLuxWrapperLayer{:prepross}
    prepross
    sentry_size::Int
end

function (f::FixedSizeDeepSet)(
    batched_input::AbstractArray,
    ps::NamedTuple,
    st::NamedTuple
)
    @assert size(batched_input)[end-1] == f.sentry_size "got $(size(batched_input)) which is incorrect, expected dim $(f.sentry_size) for axe $(ndims(batched_input) -1)"
    batched_input = reshape(batched_input, size(batched_input)[begin:end-2]..., :)
    res, st = Lux.apply(f.prepross, batched_input, ps, st)
    res = reshape(res, size(res)[begin:end-1]..., f.sentry_size, :)
    reshape(sum(res; dims=ndims(res) - 1), size(res)[begin:end-2]..., :), st
end

@inbounds function _preprocessing!(
    dot::AbstractVector{T},
    r_s::AbstractVector{T},
    r_d::AbstractVector{T},
    d_s::AbstractVector{T},
    d_d::AbstractVector{T},
    coeff::AbstractVector{T},
    r::AbstractVector{T},
    _center::AbstractVector{Point3{T}},
    distances::AbstractVector{T},
    p1::Integer,
    p2::Integer,
    i::Integer,
    batch_offset::Integer,
    cutoff_radius::T) where {T}
    _p1 =  p1 + batch_offset
    _p2 =  p2 + batch_offset
    d_1 = distances[_p1]
    d_2 = distances[_p2]
    r_1 = r[_p1]
    r_2 = r[_p2]
    r_s[i] = r_1 + r_2
    r_d[i] = abs(r_1 - r_2)
    d_s[i] = d_1 + d_2
    d_d[i] = abs(d_1 - d_2)
    c1  = _center[_p1]
    c2 = _center[_p2]
    d = zero(eltype(dot))
    for j in 1:3
        d += c1[j] * c2[j]
    end
    d += 1.0f-8
    d /= d_1 * d_2 + 1.0f-8
    dot[i] = d
    coeff[i] = cut(cutoff_radius, d_1) * cut(cutoff_radius, d_2)
end
function preprocessing!(ret::AbstractMatrix{T}, point::Point3{T}, atoms::StructVector{Sphere{T}}; cutoff_radius::T) where {T}
    (; r, center) = atoms
    _center = MallocArray{Point3{T}}(undef, length(atoms))
    distances = MallocArray{T}(undef, length(atoms))
    try
        dot = @view ret[1, :]
        r_s = @view ret[2, :]
        r_d = @view ret[3, :]
        d_s = @view ret[4, :]
        d_d = @view ret[5, :]
        coeff = @view ret[6, :]
        _center .= center .- point
        distances .= euclidean.(Ref(zero(point)), _center)
        i = 1
        for p1 in 1:length(atoms)
            for p2 in 1:p1
                _preprocessing!(dot,r_s,r_d,d_s,d_d,coeff,r,_center,distances,p1,p2,i,0,cutoff_radius)
                @assert !any(isnan.(ret[i]))
                i += 1
            end
        end
        # @assert length(dot) == i-1 "got length of $(length(dot)) instead of $(i -1)"
        # @assert !(any(isnan.(ret))) "nan values are $(findall(isnan, ret)), $(size(ret))"
    finally
        free(distances)
        free(_center)
    end
end
function nb_features(nb_atoms::T)::T where T<: Integer
     (nb_atoms* (nb_atoms+ 1)) ÷ 2
end
function get_batch_lengths(field::AbstractVector{<:AbstractVector})::Vector{Int}
    # @assert length(field) >= 1
    lengths = zeros(Int, length(field) + 1)
    for i in eachindex(field)
        l = length(field[i])
        s = nb_features(l)
        lengths[i+1] = lengths[i] + s
    end
    lengths
end
function preprocessing(
    point::Batch{<:AbstractVector{Point3{T}}},
    atoms::Batch{<:Vector{<:StructVector{Sphere{T}}}};
    cutoff_radius::T) where {T}
    lengths = get_batch_lengths(atoms.field)
    # @assert all(lengths .==vcat([0],cumsum(atoms.field .|> size .|> last .|> last |>Map(x -> x * (x + 1) ÷ 2))))
    length_tot = last(lengths)
    ret = Matrix{T}(undef, 6, length_tot)
    ret .= T(NaN)
    # Folds.foreach(eachindex(atoms.field)) do i
    @threads for i in eachindex(atoms.field)
        # if length(get_slice(lengths,i)) == 0
            # break
        # end
        preprocessing!(
            view(ret, :, get_slice(lengths, i)),
            point.field[i], atoms.field[i];
            cutoff_radius
        )
    end
    @assert !(any(isnan.(ret))) "nan values are $(findall(isnan, ret)), $(size(ret))"
    ConcatenatedBatch(ret, lengths)
end

function preprocessing(
    point::Batch{<:AbstractVector{Point3{T}}},
    atoms::Batch{<:Vector{<:StructVector{Sphere{T}}}},
    max_nb_atoms::Int;
    cutoff_radius::T) where {T}
    slice_length =nb_features(max_nb_atoms) 
    length_tot = length(atoms.field) * slice_length
    ret = zeros(T, 6, length_tot)
    # Folds.foreach(eachindex(atoms.field)) do i
    for i in eachindex(atoms.field)
        f = atoms.field[i]
        first_atoms = @view f[begin:min(length(f), max_nb_atoms)]
        preprocessing!(
            view(ret, :, (1+(i-1)*slice_length):(i*slice_length)),
            point.field[i], first_atoms;
            cutoff_radius
        )
    end
    @assert !(any(isnan.(ret)))
    reshape(ret, 6, slice_length, :)
end

function cut(cut_radius::T, r::T)::T where {T<:Number}
    k = r / cut_radius
    threshold = T(.5)
    ifelse(zero(T) <= k <= threshold, one(T), zero(T)) + ifelse(threshold < k <= one(T),(one(T) - k)/(one(T) - threshold), zero(T))
end

scale_factor(x) = @view x[end:end, :]

function trace(message::String, x)
    @info message Ref(abs.(x)) .|> [minimum, mean, std, maximum,size]
    x
end

trace(message::String) = x -> trace(message, x)
function ChainRulesCore.rrule(::typeof(trace), message, x)
    y = trace(message, x)
    function trace_pullback(y_hat)
        @info "derivation $message" Ref(abs.(y_hat)) .|> [minimum, mean, std, maximum,size]
        NoTangent(), NoTangent(), y_hat
    end
    return y, trace_pullback
end

@inline function select_neighboord(
    point::Point, grid::RegularGrid{T}
)::StructVector{Sphere{T}} where {T}
    _inrange(StructVector{Sphere{T}}, grid, point)
end

struct PreprocessingLayer <: Lux.AbstractLuxLayer
    fun::Function
end

Lux.initialparameters(::AbstractRNG, ::PreprocessingLayer) = (;)
Lux.initialstates(::AbstractRNG, ::PreprocessingLayer) = (;)
(fun::PreprocessingLayer)(arg, _, st) = fun(arg), st
((; fun)::PreprocessingLayer)(arg,) =
    ignore_derivatives() do
        fun(arg)
    end

function is_in_van_der_waals(array::AbstractArray)
    any(axes(array, 2)) do i
        array[4, i] < array[2, i] ||
            array[5, i] < array[3, i]
    end
end
function is_in_van_der_waals(b::ConcatenatedBatch)
    reshape(map(1:(length(b.lengths)-1)) do i
            is_in_van_der_waals(get_element(b, i))
        end, 1, :)
end

@concrete terse struct FunctionalLayer <: Lux.AbstractLuxContainerLayer{(:layer)}
    fun::Function
    layer
end

((; fun, layer)::FunctionalLayer)(arg, ps, st) = fun(layer, arg, ps, st)
