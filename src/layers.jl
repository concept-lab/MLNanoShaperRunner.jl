using Lux
using Adapt: @adapt_structure
using ConcreteStructs
using GeometryBasics
using Random
using SimpleChains: static
using Adapt
using StructArrays
using Distances
using ChainRulesCore
using Statistics
using Folds
using Transducers
using Static
using StaticTools
using CUDA

function terse end

"""
	ModelInput

input of the model
# Fields
- point::Point3, the position of the input
- atoms::StructVector{Sphere}, the atoms in the neighboord
"""
struct ModelInput{T <: Number}
    point::Point3{T}
    atoms::StructVector{Sphere{T}} #Set
end

struct PreprocessedData{T <: Number}
    dot::T
    r_1::T
    r_2::T
    d_1::T
    d_2::T
end
PreprocessedData(x::Vector) = PreprocessedData(map(1:5) do f
    getindex.(x, f)
end...)

#enable running on gpu
Adapt.@adapt_structure Batch
Adapt.@adapt_structure ModelInput
Adapt.@adapt_structure PreprocessedData
# Adapt.@adapt_structure Partial
@concrete terse struct DeepSet <: Lux.AbstractLuxWrapperLayer{:prepross}
    prepross
end

function (f::MLNanoShaperRunner.DeepSet)(
        (; field, lengths)::ConcatenatedBatch,
        ps::NamedTuple,
        st::NamedTuple
)
    res, st = Lux.apply(f.prepross, field, ps, st)
    batched_sum(res, lengths),st 
end
function (f::MLNanoShaperRunner.DeepSet)(arg::Batch, ps, st)
    f(ConcatenatedBatch(arg), ps, st)
end
function (f::DeepSet)(set::AbstractArray, ps, st)
    f(Batch([set]), ps, st)
end
function _make_id_product!(a::AbstractVector{T}, b::AbstractVector{T}, n::Integer) where {T}
    k = 1
    for i in 1:n
        for j in 1:i
            a[k] = i
            b[k] = j
            k += 1
        end
    end
end

function make_id_product(f, n::Integer)
    MallocArray(Int, 2, n * (n + 1) ÷ 2) do m
        a = view(m, 1, :)
        b = view(m, 2, :)
        _make_id_product!(a, b, n)
        f(a, b)
    end
end

function preprocessing!(ret,point::Point3{T}, atoms::StructVector{Sphere{T}};cutoff_radius::T) where {T}
    (; r, center) = atoms
    n = size(ret,2)
    d_1 = MallocArray{T}(undef,n)
    d_2 = MallocArray{T}(undef,n)
    r_1 = MallocArray{T}(undef,n)
    r_2 = MallocArray{T}(undef,n)
    _center = MallocArray{Point3{T}}(undef,length(atoms))
    distances = MallocArray{T}(undef,length(atoms))
    dot= @view ret[1,:]
    r_s= @view ret[2,:]
    r_d= @view ret[3,:]
    d_s= @view ret[4,:]
    d_d= @view ret[5,:]
    coeff= @view ret[6,:]

    try 
        _center .= center .- point
        distances .= euclidean.(Ref(zero(point)),_center)
        make_id_product(length(atoms)) do prod_1, prod_2
            d_1 .= distances[prod_1]
            r_1 .= r[prod_1]
            d_2 .= distances[prod_2]
            r_2 .= r[prod_2]
            d_s .= d_1 .+ d_2
            d_d .= abs.(d_1 .- d_2)
            r_s .= r_1 .+ r_2
            r_d .= abs.(r_1 .- r_2)
            dot .= (_center[prod_1] .⋅ _center[prod_2] .+ 1f-8) ./ (d_1 .* d_2 .+ 1.0f-8)
            coeff .=  cut.(cutoff_radius, d_1) .* cut.(cutoff_radius, d_2)
    end
    finally
        free(d_1)
        free(d_2)
        free(r_1)
        free(r_2)
    end
end

function preprocessing(point::Batch{Vector{Point3{T}}},
        atoms::Batch{<:Vector{<:StructVector{Sphere{T}}}};cutoff_radius::T) where {T}
    lengths = vcat(
        [0], atoms.field .|> size .|> last |> Map(x -> x * (x + 1) ÷ 2) |> cumsum)
    length_tot = last(lengths)
    ret = Matrix{T}(undef,6,length_tot)
    # Folds.foreach(eachindex(atoms.field)) do i
    foreach(eachindex(atoms.field)) do i
        preprocessing!(
            view(ret,:, get_slice(lengths, i)),
            point.field[i], atoms.field[i];
            cutoff_radius
        )
    end
    ConcatenatedBatch(ret, lengths)
end

function cut(cut_radius::T, r::T)::T where {T <: Number}
    k = r/cut_radius
    ifelse(0 <= k <= .5, one(T),zero(T)) + ifelse(.5< k <=1 , 2*(1-k),zero(T))
end

function symetrise(val::StructArray{PreprocessedData{T}};
        cutoff_radius::T) where {T <: Number}
    dot = val.dot
    d_1 = val.d_1
    d_2 = val.d_2
    r_1 = val.r_1
    r_2 = val.r_2
    vcat(dot,
        r_1 .+ r_2,
        abs.(r_1 .- r_2),
        d_1 .+ d_2, abs.(d_1 .- d_2),
        cut.(cutoff_radius, d_1) .* cut.(cutoff_radius, d_2))
end

function symetrise(val::ConcatenatedBatch{<:StructArray{<:PreprocessedData}}; kargs...)
    ConcatenatedBatch(symetrise(val.field; kargs...), val.lengths)
end

function symetrise(; cutoff_radius::Number, device)
    Partial(symetrise; cutoff_radius, device)
end

scale_factor(x) = x[end:end, :]

function trace(message::String, x)
    @debug message Ref(abs.(x)) .|> [minimum,mean,std,maximum]
    x
end

trace(message::String) = x -> trace(message, x)
function ChainRulesCore.rrule(::typeof(trace), message, x)
    y = trace(message, x)
    function trace_pullback(y_hat)
        @debug "derivation $message" Ref(abs.(y_hat)) .|> [minimum,mean,std,maximum]
        NoTangent(), NoTangent(), y_hat
    end
    return y, trace_pullback
end

"""
	AnnotedKDTree(data::StructVector,property::StaticSymbol)
# Fields
- data::StructVector
- tree::KDTree
"""
struct AnnotedKDTree{Type, Property, Subtype}
    data::StructVector{Type}
    tree::KDTree{Subtype}
    function AnnotedKDTree(data::StructVector, property::StaticSymbol)
        new{eltype(data), dynamic(property),
            eltype(getproperty(StructArrays.components(data), dynamic(property)))}(
            data, KDTree(getproperty(data, dynamic(property)); reorder = false))
    end
end

@inline function select_neighboord(
        point::Point, (; data, tree)::AnnotedKDTree{Type};
        cutoff_radius)::StructVector{Type} where {Type}
    data[inrange(tree, point, cutoff_radius)]
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
    any(axes(array,2)) do i
        array[4,i] < array[2,i] ||
        array[5,i] < array[3,i]
    end
end
function is_in_van_der_waals(b::ConcatenatedBatch)
    reshape(map(1:(length(b.lengths) - 1)) do i
            is_in_van_der_waals(get_element(b, i))
        end, 1, :)
end

@concrete terse struct FunctionalLayer <: Lux.AbstractLuxContainerLayer{(:layer)}
    fun::Function
    layer
end

((; fun, layer)::FunctionalLayer)(arg, ps, st) = fun(layer, arg, ps, st)
