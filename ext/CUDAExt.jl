module CUDAExt
using CUDA, MLNanoShaperRunner, GeometryBasics, StructArrays
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
    for j in (nb_elements[n]+1):nb_elements[n+1]
        a[i, n] += b[i, j]
    end
end

function batched_sum!(a::CuMatrix, b::CuMatrix, nb_elements::CuVector{Int})
    nb_computations = size(b, 1) * (length(nb_elements) - 1)
    block_size = 16
    @cuda threads = block_size blocks = cld(nb_computations, block_size) _kernel_sum!(
        a, b, nb_elements)
end
function MLNanoShaperRunner.batched_sum(b::CuMatrix,_, nb_elements::CuVector)
    a = similar(b, eltype(b), (size(b, 1), length(nb_elements) - 1))
    batched_sum!(a, b, nb_elements)
    a
end

function MLNanoShaperRunner.batched_sum(b::CuMatrix,i, nb_elements::AbstractVector)
    MLNanoShaperRunner.batched_sum(b,i,cu(nb_elements))
end

@inbounds function _kernel_centers_distances!(
    center::CuDeviceVector{Point3{T}},
    distances::CuDeviceVector{T},
    points::CuDeviceVector{Point3{T}},
    lengths::CuDeviceVector{Int32}) where {T}

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = blockIdx().y
    i = i + lengths[n]
    if i >= lengths[n+1]
        return
    end
    center[i] = center[i] .- points[n]
    distances[i] = sqrt( sum(center[i] .^ 2))
    return
end

function centers_distances!(
    center::CuVector{Point3{T}},
    distances::CuVector{T},
    points::CuVector{Point3{T}},
    max_set_size::Int32,
    lengths::CuVector{Int32}
) where {T}
    @cuda threads = (16, 1) blocks = (cld(max_set_size, 16), length(lengths) - 1) _kernel_centers_distances!(center, distances, points, lengths)
end

function _kernel_preprocessing!(
    dot::AbstractVector{T},
    r_s::AbstractVector{T},
    r_d::AbstractVector{T},
    d_s::AbstractVector{T},
    d_d::AbstractVector{T},
    coeff::AbstractVector{T},
    r::AbstractVector{T},
    lengths::AbstractVector{Int32},
    center::AbstractVector{Point3{T}},
    distances::AbstractVector{T},
    cutoff_radius::T) where {T}
    p1 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    p2 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    batch_dim = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    if p1 > p2 || batch_dim >= length(lengths)
        return
    end
    i = p1 + p2 * (p2 - 1) ÷ 2 + lengths[batch_dim] 
    batch_dim1 = batch_dim + one(batch_dim)
    set_size = lengths[batch_dim1]
    if i > set_size
        return
    end
    @cushow (i,p1,p2,set_size)
    # @assert i + length[batch_dim] <= length(dot)
    MLNanoShaperRunner._preprocessing!(dot, r_s, r_d, d_s, d_d, r, coeff, center, distances, p1, p2, i, cutoff_radius)
    return
end

function preprocessing!(ret::CuMatrix{T}, points::CuVector{Point3{T}},atoms::ConcatenatedBatch{<:StructVector{Sphere{T}}}, lengths::Vector{Int32}; cutoff_radius::T) where {T}
    center = atoms.field.center
    r = atoms.field.r
    distances = similar(center, T)
    centers_distances!(center, distances, points,atoms.max_set_size |> Int32, atoms.lengths .|> Int32)
    @info "post" center distances
    max_set_size = maximum(1:(length(lengths)-1)) do i
        lengths[i+1] - lengths[i]
    end
    dot = @view ret[1, :]
    r_s = @view ret[2, :]
    r_d = @view ret[3, :]
    d_s = @view ret[4, :]
    d_d = @view ret[5, :]
    coeff = @view ret[6, :]
    # @info "launching kernel" ret  lengths max_set_size
    @cuda threads = (16, 16, 1) blocks = cld.((max_set_size, max_set_size,16*length(points)), 16) _kernel_preprocessing!(dot, r_s, r_d, d_s, d_d, coeff, r, cu(lengths), center, distances, cutoff_radius)
    # @info "after kernel" size(ret)
    # @info "after kernel" ret
end

function get_batch_lengths(arglengths::AbstractVector{Int})::Vector{Int}
    lengths = zeros(Int,length(arglengths))
    for i in eachindex(lengths)[begin:(end -1)]
        lengths[i+1] = lengths[i] +  MLNanoShaperRunner.nb_features(arglengths[i+1] - arglengths[i])
    end
    lengths
end

function MLNanoShaperRunner.preprocessing(
    points::Batch{<:CuVector{Point3{T}}},
    atoms::ConcatenatedBatch{<:StructVector{Sphere{T}}};
    cutoff_radius::T) where {T}
    @assert length(points.field) == (length(atoms.lengths) - 1)
    @info "atoms lengths" atoms.lengths
    sets_cum_lengths = get_batch_lengths(atoms.lengths)
    atoms = cu(atoms)
    length_tot = last(sets_cum_lengths)
    ret = similar(points.field, T, 6, length_tot)
    ret .= -1
    preprocessing!(ret, points.field, atoms, sets_cum_lengths .|> Int32; cutoff_radius)
    @info "end" Array(ret)
    ConcatenatedBatch(ret, sets_cum_lengths)
end
@inline function MLNanoShaperRunner.preprocessing(
    points::Batch{<:CuVector{Point3{T}}},
    atoms::Batch{<:Vector{<:StructVector{Sphere{T}}}};
    cutoff_radius::T
) where {T}
    MLNanoShaperRunner.preprocessing(
        points, ConcatenatedBatch(atoms); cutoff_radius
    )
end
function _kernel_batched_sum_pullback(delta_b::CuDeviceMatrix{T},delta::CuDeviceMatrix{T},nb_elements::CuDeviceVector{<:Integer}) where T <: Real
    i = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    k = threadIdx().z + (blockIdx().z - 1) * blockDim().z
    if i+1  > length(nb_elements)
        return
    end
    if k > size(delta_b,1)
        return
    end
    j = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j += nb_elements[i]
    if j > nb_elements[i + one(i)]
        return
    end
    delta_b[k, j] = delta[k, i]
    return
end
function MLNanoShaperRunner.batched_sum_pullback(b_sum::CuMatrix{T},delta::CuMatrix{T},max_set_size::Integer,nb_elements::CuVector{<:Integer})::CuMatrix{T} where T
    delta_b = similar(b_sum)
    @cuda threads = (16, 16, 1) blocks = cld.((length(nb_elements) - 1 , max_set_size,16* size(b_sum,1)),16) _kernel_batched_sum_pullback(delta_b,delta,nb_elements)
    delta_b
end
function MLNanoShaperRunner.batched_sum_pullback(b_sum::CuMatrix{T},delta::CuMatrix{T},max_set_size::Integer,nb_elements::AbstractVector{<:Integer})::CuMatrix{T} where T
    MLNanoShaperRunner.batched_sum_pullback(b_sum,delta,max_set_size,cu(nb_elements))
end
end
