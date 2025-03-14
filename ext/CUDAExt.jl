module CUDAExt
using CUDA, MLNanoShaperRunner
function _kernel_sum!(a::CuDeviceMatrix{T}, b::CuDeviceMatrix{T},
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
    block_size = 1024
    @cuda threads=block_size blocks=(1 + (nb_computations - 1) ÷ block_size) _kernel_sum!(
        a, b, nb_elements)
end
function MLNanoShaperRunner.batched_sum(b::CuMatrix, nb_elements::AbstractVector)
    a = similar(b, eltype(b), (size(b, 1), length(nb_elements) - 1))
    batched_sum!(a, b, cu(nb_elements))
    a
end
end
