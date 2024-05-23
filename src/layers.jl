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
using Static

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

struct PreprocessData{T<:AbstractArray{<:Number}}
    dot::T
    r_1::T
    r_2::T
    d_1::T
    d_2::T
end

#enable running on gpu
Adapt.@adapt_structure Batch
Adapt.@adapt_structure ModelInput
Adapt.@adapt_structure PreprocessData


PreprocessData(x::Vector) = PreprocessData(map(1:5) do f
    getindex.(x, f)
end...)

function preprocessing((; point, atoms)::ModelInput)
    prod = reduce(vcat, map(eachindex(atoms)) do i
        map(1:i) do j
            atoms[i], atoms[j]
        end
    end)
    x = map(prod) do (atom1, atom2)::Tuple{Sphere,Sphere}
        d_1 = euclidean(point, atom1.center)
        d_2 = euclidean(point, atom2.center)
        dot = (atom1.center - point) ⋅ (atom2.center - point) / (d_1 * d_2 + 1.0f-8)
        (dot, atom1.r, atom2.r, d_1, d_2)
    end |> vec
    PreprocessData(map(1:5) do f
        reshape(getfield.(x, f), 1, :)
    end...)
end

preprocessing(x::Batch) = Batch(preprocessing.(x.field))

expand_dims(x::AbstractArray, n::Integer) = reshape(x, size(x)[begin:n-1]..., n, size(x)[n:end]...)

@concrete struct DeepSet <: Lux.AbstractExplicitContainerLayer{(:prepross,)}
    prepross
end

function (f::DeepSet)(set::AbstractVector{<:AbstractArray}, ps, st)
    trace("input size", length(set))
    sum(set) do arg
        Lux.apply(f.prepross, arg, ps, st) |> first
    end, st#/ sqrt(length(set)), st
end
function (f::DeepSet)(arg::PreprocessData, ps, st)
    trace("input size", length(arg.dot))
    sum(Lux.apply(f.prepross, arg, ps, st) |> first), st # / sqrt(length(arg.dot)), st
end
function (f::DeepSet)(arg::Batch{<:AbstractVector{<:PreprocessData}}, ps, st)
    lengths = vcat([0], getfield.(arg.field, :dot) .|> size .|> last |> cumsum)
    batched = PreprocessData(map(fieldnames(PreprocessData)) do i
        cat(getfield.(arg.field, i)...; dims=ndims(first(arg.field).dot))
    end...)
    res = Lux.apply(f.prepross, batched, ps, st) |> first |> trace("raw")
    mapreduce(hcat, 1:(length(lengths)-1)) do i
        sum(res[:, (lengths[i]+1):lengths[i+1]]; dims=2)
    end, st
end

function symetrise((; dot, r_1, r_2, d_1, d_2)::PreprocessData{<:AbstractArray{T}}; cutoff_radius::T) where {T<:Number}
    cutoff_radius::Float32
    res = vcat(dot, r_1 .+ r_2, abs.(r_1 .- r_2), d_1 .+ d_2, abs.(d_1 .- d_2))
    res .* cut.(cutoff_radius, r_1) .* cut.(cutoff_radius, r_2)
end

struct Partial{F<:Function,A<:Tuple,B<:Base.Pairs}<:Function
    f::F
    args::A
    kargs::B
	Partial(f, args...; kargs...) = new{typeof(f),typeof(args),typeof(kargs)}(f,args,kargs)
end
(f::Partial)(args...;kargs...) = f.f(f.args...,args...;f.kargs...,kargs...)

symetrise(; cutoff_radius::Number) = Partial(symetrise;cutoff_radius) |> Lux.WrappedFunction

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

function (l::Encoding{T})(input::PreprocessData{<:AbstractArray{T}},
    (; dotsₛ, η, ζ, Dₛ),
    st) where {T}
    (; dot, d_1, d_2, r_1, r_2) = input |> trace("input")
    encoded = ((2 .+ dot .- tanh.(dotsₛ)) ./ 4) .^ ζ .*
              exp.(-η .* ((d_1 .+ d_2) ./ 2 .- Dₛ) .^ 2) .*
              cut.(l.cut_distance, d_1) .*
              cut.(l.cut_distance, d_2)
    res = vcat(map((encoded, (r_1 .+ r_2) ./ 2, abs.(r_1 .- r_2))) do x
        mergedims(x, 1:2)
    end...)
    res |> trace("features"), st
end

function cut(cut_radius::T, r::T)::T where {T<:Number}
    ifelse(r >= cut_radius, zero(T), (1 + cos(π * r / cut_radius)) / 2)
end

function preoprocessed_reshape(x::PreprocessData, arg::Union{Int,Colon}...)
    PreprocessData(map(fieldnames(PreprocessData)) do name
        reshape(getfield(x, name), arg...)
    end...)
end
preoprocessed_reshape(arg::Union{Int,Colon}...) = x -> preoprocessed_reshape(x, arg...)

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
    point, (; data, tree)::AnnotedKDTree; cutoff_radius)
    data[inrange(tree, point, cutoff_radius)]
end
# function ChainRulesCore.rrule(
#         ::typeof(Base.getproperty), array::StructArray{T}, field::Symbol) where {T}
#     member = getproperty(array, field)
#     function getproperty_pullback(y_hat)
#         (NoTangent(),
#             StructArray(;
#                 (f => if f == field
#                      y_hat
#                  else
#                      zero(getproperty(array, f))
#                  end
#                 for f in propertynames(array))...),
#             NoTangent()) |> trace("getproperty_pullback")
#     end
#     member, getproperty_pullback
# end
#
# function ChainRulesCore.rrule(::Type{StructArray}, fields::Tuple)
#     res = StructArray(fields)
#     function StructArray_pullback(df)
#     end
#     res, StructArray_pullback
# end
