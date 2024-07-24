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
using Static
using CUDA

function terse end
struct Batch{T<:Vector}
    field::T
end
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

struct PreprocessData{T<:Number}
    dot::T
    r_1::T
    r_2::T
    d_1::T
    d_2::T
end

PreprocessData(x::Vector) = PreprocessData(map(1:5) do f
    getindex.(x, f)
end...)

struct Partial{F<:Function,A<:Tuple,B<:Base.Pairs} <: Function
    f::F
    args::A
    kargs::B
    function Partial(f, args...; kargs...)
        new{typeof(f),typeof(args),typeof(kargs)}(f, args, kargs)
    end
end
(f::Partial)(args...; kargs...) = f.f(f.args..., args...; f.kargs..., kargs...)

broadcasted(f) = Partial(f, broadcast)

#enable running on gpu
Adapt.@adapt_structure Batch
Adapt.@adapt_structure ModelInput
Adapt.@adapt_structure PreprocessData
Adapt.@adapt_structure Partial
@concrete terse struct DeepSet <: Lux.AbstractExplicitContainerLayer{(:prepross,)}
    prepross
end


function (f::MLNanoShaperRunner.DeepSet)(arg::Batch, ps, st::NamedTuple)
    lengths = vcat([0], arg.field .|> size .|> last |> cumsum)
	get_slice(i) =  (lengths[i]+1:lengths[i+1])
    batched = evaluate_and_cat(length(arg.field),arg.field|> first,get_slice) do i
		arg.field[i]
	end
    res::AbstractMatrix{<:Number} = Lux.apply(f.prepross, batched, ps, st) |> first 
	batched_sum(res,lengths), st
end
function (f::DeepSet)(set::AbstractArray, ps, st)
    f(Batch([set]), ps, st)
end
# function (f::MLNanoShaperRunner.DeepSet)(arg::Batch, ps, st::NamedTuple)
#     lengths = vcat([0], arg.field .|> size .|> last |> cumsum)
#     get_slice(i) = (lengths[i]+1:lengths[i+1])
#     batched = evaluate_and_cat(length(arg.field), arg.field |> first, get_slice) do i
#         arg.field[i]
#     end
#     res::AbstractMatrix{<:Number} = Lux.apply(f.prepross, batched, ps, st) |> first |> trace("raw")
#     @assert size(res, 2) == last(lengths)
#     sub_array = @view res[:, (lengths[1]+1):lengths[2]]
#     evaluate_and_cat(length(arg.field), sub_array, i -> i:i) do i
#         sum(@view res[fill(:, ndims(sub_array) - 1)..., get_slice(i)]; dims=ndims(sub_array))
#     end, st
# end

function make_id_product(f, n)
    MallocArray(Int, 2, n * (n + 1) ÷ 2) do m
        a = view(m, 1, :)
        b = view(m, 2, :)
        _make_id_product!(a, b, n)
        f(a, b)
    end
end

function preprocessing((; point, atoms)::ModelInput{T}) where {T}
    (; r, center) = atoms
    make_id_product(length(atoms)) do prod_1, prod_2
        n_tot = length(prod_1)
        distances = euclidean.(Ref(point), center)
        d_1 = distances[prod_1]
        r_1 = r[prod_1]
        d_2 = distances[prod_2]
        r_2 = r[prod_2]
        dot = map(1:n_tot) do n
            (center[prod_1[n]] - point) ⋅ (center[prod_2[n]] - point) / (d_1[n] * d_2[n] + 1.0f-8)
        end
        res = StructArray{PreprocessData{T}}((dot, r_1, r_2, d_1, d_2))
        reshape(res, 1, :)
    end
end

function preprocessing((; point, atoms)::Tuple{Batch{Point3{T}},StructVector{Sphere{T}}}) where {T}
    Folds.map(point) do point
        preprocessing(ModelInput(point, atoms))
    end |> Batch
end

function cut(cut_radius::T, r::T)::T where {T<:Number}
    ifelse(r >= cut_radius, zero(T), (1 + cos(π * r / cut_radius)) / 2)
end

function symetrise((; dot, r_1, r_2, d_1, d_2)::StructArray{PreprocessData{T}};
    cutoff_radius::T) where {T<:Number}
    vcat(dot,
        r_1 .+ r_2,
        abs.(r_1 .- r_2),
        d_1 .+ d_2, abs.(d_1 .- d_2)) .*
    cut.(cutoff_radius, r_1) .* cut.(cutoff_radius, r_2)
end
scale_factor(x) = x[end:end, :]

function symetrise(; cutoff_radius::Number)
    Partial(symetrise; cutoff_radius) |> Lux.WrappedFunction
end
function expand_dims(x::AbstractArray, n::Integer)
    reshape(x, size(x)[begin:(n-1)]..., n, size(x)[n:end]...)
end


function mergedims(x::AbstractArray, dims::AbstractRange)
    pre = size(x)[begin:(first(dims)-1)]
    merged = size(x)[dims]
    post = size(x)[(last(dims)+1):end]
    reshape(x, (pre..., prod(merged), post...))
end

function struct_stack(x::AbstractArray{PreprocessData{T}}) where {T}
    f(field) = reshape(getproperty.(x, field), 1, 1, size(x)...)
    x = StructVector{PreprocessData{T}}(f.(fieldnames(PreprocessData))) |>
        trace("pre struct array")
end

function trace(message::String, x)
    @debug message x
    x
end

trace(message::String) = x -> trace(message, x)
function ChainRulesCore.rrule(::typeof(trace), message, x)
    y = trace(message, x)
    function trace_pullback(y_hat)
        @debug "derivation $message" y_hat
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
struct AnnotedKDTree{Type,Property,Subtype}
    data::StructVector{Type}
    tree::KDTree{Subtype}
    function AnnotedKDTree(data::StructVector, property::StaticSymbol)
        new{eltype(data),dynamic(property),
            eltype(getproperty(StructArrays.components(data), dynamic(property)))}(
            data, KDTree(getproperty(data, dynamic(property)); reorder=false))
    end
end


function select_neighboord(
    point::Point, (; data, tree)::AnnotedKDTree; cutoff_radius)
    data[inrange(tree, point, cutoff_radius)]
end

struct PreprocessingLayer <: Lux.AbstractExplicitLayer
    fun::Function
end

Lux.initialparameters(::AbstractRNG, ::PreprocessingLayer) = (;)
Lux.initialstates(::AbstractRNG, ::PreprocessingLayer) = (;)
(fun::PreprocessingLayer)(arg, _, st) = fun(arg), st
((; fun)::PreprocessingLayer)(arg,) =
    ignore_derivatives() do
        fun(arg)
    end
function is_in_van_der_val((; d_1, d_2, r_1, r_2)::PreprocessData)
    d_1 <= r_1 || d_2 <= r_2
end
function is_in_van_der_val(array::AbstractArray{<:PreprocessData})
    is_in_van_der_val.(array) |> any
end
function is_in_van_der_val(b::Batch)
    reshape(map(b.field) do array
            is_in_van_der_val(array)
        end, 1, :)
end

@concrete terse struct FunctionalLayer <: Lux.AbstractExplicitContainerLayer{(:layer)}
    fun::Function
    layer
end

((; fun, layer)::FunctionalLayer)(arg, ps, st) = fun(layer, arg, ps, st)
