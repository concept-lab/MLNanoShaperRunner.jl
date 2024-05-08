using NearestNeighbors
using GeometryBasics
using LinearAlgebra
using Base.Iterators

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

distance(x::Point3, y::KDTree) = nn(y, x) |> last

function signed_distance(p::Point3, mesh::RegionMesh)
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
function distance(x::GeometryBasics.Mesh, y::KDTree)
    maximum(coordinates(x)) do x
        distance(x, y)
    end
end
