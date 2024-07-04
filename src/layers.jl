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

function (f::DeepSet)(set::AbstractArray, ps, st)
    f(Batch([set]), ps, st)
end
function (f::DeepSet)(arg::Batch, ps, st)
    lengths = vcat([0], arg.field .|> size .|> last |> cumsum)
    batched = ignore_derivatives() do
        batched = similar(
            arg.field |> first,
            arg.field |> first |> eltype,
			(size(arg.field |> first)[begin:end-1]..., size.(arg.field, arg.field |> first |> ndims) |> sum))
        Folds.map(size.(arg.field, arg.field |> first |> ndims) |> cumsum |> enumerate) do (i, offset)
			batched[fill(:, ndims(arg.field |> first) - 1)..., offset] = arg.field[i]
        end
        batched |> trace("batched")
    end
    res::AbstractMatrix{<:Number} = Lux.apply(f.prepross, batched, ps, st) |> first |> trace("raw")
    @assert size(res, 2) == last(lengths)
    mapreduce(hcat, 1:(length(lengths)-1)) do i
        sum(res[:, (lengths[i]+1):lengths[i+1]]; dims=2)
    end, st
end

function preprocessing((; point, atoms)::ModelInput{T}) where {T}
    if length(atoms) == 0
        return PreprocessData{T}[] |> StructVector
    end
    prod = reduce(vcat, map(eachindex(atoms)) do i
        map(1:i) do j
            atoms[i], atoms[j]
        end
    end)
    reshape(
        map(prod) do (atom1, atom2)::Tuple{Sphere,Sphere}
            d_1 = euclidean(point, atom1.center)
            d_2 = euclidean(point, atom2.center)
            dot = (atom1.center - point) ⋅ (atom2.center - point) / (d_1 * d_2 + 1.0f-8)
            PreprocessData(dot, atom1.r, atom2.r, d_1, d_2)
        end |> StructVector,
        1,
        :)
end

function cut(cut_radius::T, r::T)::T where {T<:Number}
    ifelse(r >= cut_radius, zero(T), (1 + cos(π * r / cut_radius)) / 2)
end

function symetrise((; dot, r_1, r_2, d_1, d_2)::StructArray{PreprocessData{T}};
    cutoff_radius::T) where {T<:Number}
    vcat(dot,
        r_1 .+ r_2,
        abs.(r_1 .- r_2),
        d_1 .+ d_2, abs.(d_1 .- d_2),
        cut.(cutoff_radius, r_1) .* cut.(cutoff_radius, r_2)
    )
end
scale_factor(x) = x[end:end, :]

function symetrise(; cutoff_radius::Number)
    Partial(symetrise; cutoff_radius) |> Lux.WrappedFunction
end
function expand_dims(x::AbstractArray, n::Integer)
    reshape(x, size(x)[begin:(n-1)]..., n, size(x)[n:end]...)
end

"""
Encoding(n_dotₛ,n_Dₛ,cut_distance)

A lux layer which embed angular and radial `PreprocessData` into a feature vector invariant by translation and rotations.

# Arguments
- `n_dotₛ`: Integer specifying the number of anguar features 
- `n_Dₛ`: Integer specifying the number of radial features 
- `cut_distance`: The maximun distance of intaraction between atoms 
# Input
- `(;dot,r_1,r_2,d_1,d_2)`:`PreprocessData`, the dot product,the atoms radii and the distances between the reference point and the atoms. 
# Output
- `x`: a `Vector` representing the encoded features:
```math
x_{ij} = (\\frac{1}{2} + \\frac{dot - dot_{si}}{4})^\\eta * \\exp(-\\zeta ~ ( \\frac{d_1 + d_2}{2} - D_{si} ) ) \\times cut(d_1) \\times cut(d_2) 
```
"""
struct Encoding{T<:Number} <: Lux.AbstractExplicitLayer
    n_dotₛ::Int
    n_Dₛ::Int
    cut_distance::T
end

function Lux.initialparameters(::AbstractRNG, l::Encoding{T}) where {T}
    (dotsₛ=reshape(collect(range(T(0), T(1); length=l.n_dotₛ)), 1, :),
        Dₛ=reshape(collect(range(T(0), l.cut_distance; length=l.n_Dₛ)), :, 1),
        η=ones(T, 1, 1) ./ l.n_dotₛ,
        ζ=ones(T, 1, 1) ./ l.n_Dₛ)
end
Lux.initialstates(::AbstractRNG, l::Encoding) = (;)

function mergedims(x::AbstractArray, dims::AbstractRange)
    pre = size(x)[begin:(first(dims)-1)]
    merged = size(x)[dims]
    post = size(x)[(last(dims)+1):end]
    reshape(x, (pre..., prod(merged), post...))
end

function (l::Encoding{T})(input::StructVector{PreprocessData{T}},
    (; dotsₛ, η, ζ, Dₛ),
    st) where {T}
    (; dot, d_1, d_2, r_1, r_2) = input |> trace("input")
    encoded = ((2 .+ dot .- tanh.(dotsₛ)) ./ 4) .^ ζ .*
              exp.(-η .* ((d_1 .+ d_2) ./ 2 .- Dₛ) .^ 2) .*
              cut.(l.cut_distance, d_1) .*
              cut.(l.cut_distance, d_2)
    res::AbstractMatrix{<:Number} = vcat(map((encoded, (r_1 .+ r_2) ./ 2, abs.(r_1 .- r_2))) do x
        mergedims(x, 1:2)
    end...)
    res |> trace("features"), st
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
