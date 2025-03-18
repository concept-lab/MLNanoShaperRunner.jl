using NearestNeighbors
using GeometryBasics
using LinearAlgebra
using ChainRulesCore
using Base.Iterators
using StructArrays

struct RegionMesh
    triangles::Vector{TriangleFace{Point3f}}
    tree::KDTree
end
function RegionMesh(mesh::GeometryBasics.Mesh)
    triangles = Vector{TriangleFace{Point3f}}(undef, length(coordinates(mesh)))
    for (j, tri) in enumerate(faces(mesh))
        ctri = map(tri) do j
            coordinates(mesh)[j]
        end
        for i in tri
            triangles[i] = ctri
        end
    end
    RegionMesh(triangles,
        KDTree(coordinates(mesh); reorder = false))
end

distance(x::AbstractVector, y::KDTree)::Number = nn(y, x) |> last

"""
    signed_distance(p::Point3, mesh::RegionMesh)::Number

returns the signed distance between point p and the mesh
a positive distance means that we are inside the mesh
"""
function signed_distance(p::Point3{T}, mesh::RegionMesh)::T where {T <: Number}
    id_point, dist = nn(mesh.tree, p)
    x, y, z = mesh.triangles[OffsetInteger{-1, UInt32}(id_point)]
    # @info "triangle" x y z

    direction = hcat(y - x, z - x, p - x) |> det |> sign
    -direction * dist
end

nograd(f, args...; kargs...) = f(args...; kargs...)

function ChainRulesCore.rrule(::typeof(nograd), f, args...; kargs...)
    res = f(args...; kargs...)
    function knn_pullback(_)
        tuple(fill(NoTangent(), length(args)))
    end
    res, knn_pullback
end
"""
    distance(x::GeometryBasics.Mesh, y::KDTree)

Return the Hausdorff distance betwen the mesh coordinates
"""
function distance(vec::AbstractVector{<:AbstractVector}, y::KDTree)::Number
    minimum(vec) do x
        distance(x, y)
    end
end

struct RegularGrid{T}
    grid::Array{Vector{Sphere{T}},3}
    radius::T
    start::Point3{T}
end

function RegularGrid(points::StructVector{Sphere{T}},radius::T) where T
    mins = map(1:3) do k minimum(p -> p[k],points.center) end
    maxes = map(1:3) do k maximum(p -> p[k],points.center) end
    start = Point3(mins .- radius...)
    x_m,y_m,z_m = floor.(Int,(maxes .- mins) ./ radius .+ 1)
    pos  = map(points.center) do point floor.(Int, (point .- start) ./ radius) end 
    grid = [Sphere{T}[] for _ in 1:x_m, _ in 1:y_m, _ in 1:z_m]
    for i in eachindex(points)
        push!(grid[pos[i]...],points[i])
    end
    RegularGrid(grid,radius,start)
end

function _inrange(g::RegularGrid{T},p::Point3{T}) where T
    x,y,z = floor.(Int, (p .- g.start) ./ g.radius)
    dx = [-1,-1,-1, -1,-1,-1, -1,-1,-1,  0, 0, 0,  0,0,0,  0,0,0,  1, 1, 1,  1,1,1,  1,1,1]
    dy = [-1,-1,-1,  0, 0, 0,  1, 1, 1, -1,-1,-1,  0,0,0,  1,1,1, -1,-1,-1,  0,0,0,  1,1,1]
    dz = [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1,0,1, -1,0,1, -1, 0, 1, -1,0,1, -1,0,1]
    res = Sphere{T}[]
    for i in 1:27
        x1 = x + dx[i]
        y1 = y + dy[i]
        z1 = z + dz[i]
        if x1 in axes(g.grid,1) && y1 in axes(g.grid,2) && z1 in axes(g.grid,3)
            for s in g.grid[x1,y1,z1]
                if sum((p .- s.center) .^2) < g.radius^2
                    push!(res,s)
                end
            end
        end
    end
    res
end
