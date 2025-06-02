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
    mins = map(1:3) do k
        minimum(p -> p[k], center(points))
    end
    maxes =Point3f(map(1:3) do k
        maximum(p -> p[k], center(points))
    end...)
    start = Point3(mins...)
    x_m, y_m, z_m = get_id(maxes, start, radius)
    pos = map(center(points)) do point
        get_id(point, start, radius)
    end
    grid = [G[] for _ in 1:x_m, _ in 1:y_m, _ in 1:z_m]
    for i in eachindex(points)
        push!(grid[pos[i]...], points[i])
    end
    RegularGrid(grid, radius, start, center)
end

const dx = @SVector [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
const dy = @SVector [-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1]
const dz = @SVector [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]

function __inrange(f!::Function, g::RegularGrid{T}, p::Point3{T}) where {T}
    x, y, z = get_id(p, g.start, g.radius)
    r2 = g.radius^2
    for i in 1:27
        x1 = x + dx[i]
        y1 = y + dy[i]
        z1 = z + dz[i]
        if x1 in axes(g.grid, 1) && y1 in axes(g.grid, 2) && z1 in axes(g.grid, 3)
            for s in g.grid[x1, y1, z1]
                if sum((p .- g.center(s)) .^ 2) < r2
                    f!(s)
                end
            end
        end
    end
end
function _inrange(::Type{G}, g::RegularGrid{T}, p::Point3{T})::G where {T,G}
    res = _summon_type(G)(undef, 0)
    __inrange(x -> push!(res, x), g, p)
    res
end

function my_push!(x::AbstractMatrix{T},i::Ref{Int},j::Int, y::T) where {T}
    x[i[],j] = y
    i[] += 1
end
_sub_array_type(::Type{T}) where T <: AbstractMatrix=SubArray{eltype(T),1,T,Tuple{UnitRange{Int64},Int},true}
_sub_array_type(::Type{T}) where T <: StructArray = StructArray{eltype(T),NamedTuple,1,T,Tuple{UnitRange{Int64},Int},true}
function _inrange(::Type{G}, g::RegularGrid{T}, p::Batch{<:AbstractVector{Point3{T}}}) where {T,G}
    n = length(p.field)
    res = _summon_type(G)(undef, 128, n)
    ret = Vector{SubArray{eltype(G),1,G,Tuple{UnitRange{Int64},Int},true}}(undef, n)
    i = Ref(1)
    for j in 1:n
        i[] = 1
        __inrange(x -> my_push!(res,i,j,x), g, p.field[j],dx,dy,dz)
        ret[j] =@view  res[1:i[],j]
    end
    ret
end
