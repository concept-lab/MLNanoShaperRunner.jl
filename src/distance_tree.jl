using GeometryBasics
using LinearAlgebra
using ChainRulesCore
using Base.Iterators
using StructArrays
struct RegularGrid{T<:Number,G,F<:Function}
    grid::Array{Vector{G},3}
    radius::T
    start::Point3{T}
    center::F
end

@inline function get_id(point::Point3{T}, start::Point3{T}, radius::T)::Point3{Int} where T
    factor =  (point .- start) ./ radius
    Point3(floor(Int,factor[1]) +1,floor(Int,factor[2]) +1 ,floor(Int,factor[3]) +1 )
end

center(x) = x.center
_summon_type(::Type{G}) where {G<:AbstractArray} = G
_summon_type(::Type{<:StructArray{T}}) where {T} = StructArray{T}

function RegularGrid(points::AbstractVector{G}, radius::T, center::Function=center) where {T,G}
    mins = Point3f(map(1:3) do k
        minimum(p -> p[k], center(points))
    end...)
    maxes =Point3f(map(1:3) do k
        maximum(p -> p[k], center(points))
    end...)
    x_m, y_m, z_m = get_id(maxes, mins, radius)
    pos = map(center(points)) do point
        get_id(point, mins, radius)
    end
    grid = [G[] for _ in 1:x_m, _ in 1:y_m, _ in 1:z_m]
    for i in eachindex(points)
        push!(grid[pos[i]...], points[i])
    end
    RegularGrid(grid, radius, mins, center)
end

const dx = @SVector [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
const dy = @SVector [-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1]
const dz = @SVector [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]
const Δ3 = CartesianIndices((3,3,3)) .- CartesianIndex((2,2,2))
function _iter_grid(f!::Function,g::RegularGrid{T},p::Point3{T},Δ::CartesianIndices{3}) where {T}
    x, y, z = get_id(p, g.start, g.radius)
    for d in Δ
        x1 = x + d[1]
        y1 = y + d[2]
        z1 = z + d[3]
        if x1 in axes(g.grid, 1) && y1 in axes(g.grid, 2) && z1 in axes(g.grid, 3)
            for s in g.grid[x1, y1, z1]
                if f!(s)
                    break
                end
            end
        end
    end
end
function __inrange(f!::Function, g::RegularGrid{T}, p::Point3{T}) where {T}
    r2 = g.radius^2
    _iter_grid(g,p,Δ3) do s
        # @info "testing point" s p  sum((p .- g.center(s)) .^ 2)
        if sum((p .- g.center(s)) .^ 2) < r2
            f!(s)
        else
            false
        end
    end
    return
end
function _inrange(::Type{G}, g::RegularGrid{T}, p::Point3{T})::G where {T,G}
    res::G = _summon_type(G)(undef, 0)
    __inrange(x -> (push!(res, x);false), g, p)
    res
end

function my_push!(x::AbstractMatrix{T},i::Ref{Int},j::Int, y::T) where {T}
    i[] += 1
    x[i[],j] = y
end
_sub_array_type(::Type{T}) where T <: AbstractArray=SubArray{eltype(T),1,Matrix{eltype(T)},Tuple{UnitRange{Int64},Int},true}
function _sub_array_type(::Type{T}) where T <: StructArray 
    elt = eltype(T)
    fields = map(fieldnames(elt)) do n
        Vector{fieldtype(elt,n)} |> _sub_array_type
    end
    StructVector{elt,NamedTuple{fieldnames(elt),Tuple{fields...}}}
end
function _inrange(::Type{G}, g::RegularGrid{T}, p::Batch{<:AbstractVector{Point3{T}}}) where {T,G}
    n = length(p.field)
    result_matrix = _summon_type(G)(undef, 128, n)
    end_indices = zeros(Int,n)
    i = Ref(0)
    @threads for j in 1:n
        i[] = 0
        __inrange(x -> (my_push!(result_matrix,i,j,x);false), g, p.field[j])
        end_indices[j] = i[]
    end
    [view(result_matrix,1:i,j) for (j,i) in enumerate(end_indices)]
end
