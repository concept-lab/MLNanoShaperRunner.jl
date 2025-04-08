module CUDAExt
using CUDA, MLNanoShaperRunner,GeometryBasics,StructArrays
@inbounds function _kernel_sum!(a::CuDeviceMatrix{T}, b::CuDeviceMatrix{T},
        nb_elements::CuDeviceVector{Int}) where {T}
    nb_lines = size(b, 1)
    identifiant = (threadIdx().x - 1) + blockDim().x * (blockIdx().x - 1)
    i, n = identifiant % nb_lines + 1, identifiant ÷ nb_lines + 1
    if n + 1 > length(nb_elements)
        # we are launching mor threads than required
        return
    end
    a[i, n] = zero(T)
    for j in (nb_elements[n] + 1):nb_elements[n + 1]
        a[i, n] += b[i, j]
    end
end

function batched_sum!(a::CuMatrix, b::CuMatrix, nb_elements::CuVector{Int})
    nb_computations = size(b, 1) * (length(nb_elements) - 1)
    block_size = 16
    @cuda threads=block_size blocks=cld(nb_computations,block_size) _kernel_sum!(
        a, b, nb_elements)
end
function MLNanoShaperRunner.batched_sum(b::CuMatrix, nb_elements::AbstractVector)
    a = similar(b, eltype(b), (size(b, 1), length(nb_elements) - 1))
    batched_sum!(a, b, cu(nb_elements))
    a
end

@inbounds function _kernel_centers_distances!(
    center::CuDeviceVector{Point3{T}},
    distances::CuDeviceVector{T},
    points::CuDeviceVector{Point3{T}},
    lengths::CuDeviceVector{Int32}) where T

    i = threadIdx().x  + (blockIdx().x-1) * blockDim().x
    n = blockIdx().y
    i = i + length[n]
    if i > lengths[n+1]
        return
    end
    center[i] .-= points[lengths[n]]
    distances[i] =sqrt(sum(center[i] .^2))
end

function centers_distances!(
    center::CuVector{Point3{T}},
    distances::CuVector{T},
    points::CuVector{Point3{T}},
    lengths::Vector{Int32}
) where T
    max_set_size = maximum(1:(length(lengths)-1)) do i
        lengths[i+1] - lengths[i]
    end
    @cuda threads = (16,1) blocks = (cld(max_set_size, 16) ,length(lengths)-1) _kernel_centers_distances!(center,distances,points,cu(lengths))
end

function _kernel_preprocessing!(
    dot::CuDeviceVector{T},
    r_s::CuDeviceVector{T},
    r_d::CuDeviceVector{T},
    d_s::CuDeviceVector{T},
    d_d::CuDeviceVector{T},
    coeff::CuDeviceVector{T},
    r::CuDeviceVector{T},
    lengths::CuDeviceVector{Int32},
    center::CuDeviceVector{Point3{T}},
    distances::CuDeviceVector{T},
    cutoff_radius::T) where T
    p1 = threadIdx().x  + (blockIdx().x-1) * blockDim().x
    p2 = threadIdx().y  + (blockIdx().y-1) * blockDim().y

    if p1 > p2
        return
    end
    i = p1 + p2 * (p2 -1) ÷ 2
    batch_dim = blockIdx().x
    MLNanoShaperRunner._preprocessing!(dot,r_s,r_d,d_s,d_d,r,coeff,center,distances,p1,p2,i + lengths[batch_dim],cutoff_radius)
end

function preprocessing!(ret::CuMatrix{T},points::CuVector{Point3{T}},r::CuVector{T},center::CuVector{Point3{T}},lengths::Vector{Int32};cutoff_radius::T) where T
    distances = similar(points,T)
    centers_distances!(center,distances,points,lengths)
    max_set_size = maximum(1:(length(lengths)-1)) do i
        lengths[i+1] - lengths[i]
    end
    dot = @view ret[1, :]
    r_s = @view ret[2, :]
    r_d = @view ret[3, :]
    d_s = @view ret[4, :]
    d_d = @view ret[5, :]
    coeff = @view ret[6, :]
    @cuda threads = (16,16,1) blocks = cld.((max_set_size,max_set_size,1),16) _kernel_preprocessing!(dot,r_s,r_d,d_s,d_d,coeff,r,lengths,center,distances,cutoff_radius)
end
function MLNanoShaperRunner.preprocessing(
    points::Batch{<:CuVector{Point3{T}}},
    atoms::ConcatenatedBatch{<:StructVector{Sphere{T}}};
    cutoff_radius::T) where {T}
    lengths = atoms.lengths
    length_tot = last(lengths)
    ret = similar(points.field,T,5, length_tot)
    preprocessing!(ret,points,atoms.r,atoms.center,lengths .|> Int32)
    ConcatenatedBatch(res,lengths)
end
end
