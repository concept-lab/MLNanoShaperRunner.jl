using CUDA
using ChainRulesCore
using Folds

struct Partial{F<:Function,A<:Tuple,B<:Base.Pairs} <: Function
    f::F
    args::A
    kargs::B
    function Partial(f, args...; kargs...)
        new{typeof(f),typeof(args),typeof(kargs)}(f, args, kargs)
    end
end
(f::Partial)(args...; kargs...) = f.f(f.args..., args...; f.kargs..., kargs...)

function _kernel_sum!(a::CuDeviceMatrix{T}, b::CuDeviceMatrix{T}, nb_elements::CuDeviceVector{Int}) where {T}
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
    block_size = 1024
    @cuda threads = block_size blocks = (1 + (nb_computations - 1) ÷ block_size) _kernel_sum!(a, b, nb_elements)
end

function batched_sum!(a::AbstractMatrix{T}, b::AbstractMatrix{T}, nb_elements::AbstractVector{Int}) where {T}
    nb_lines = size(b, 1)
    Folds.foreach(0:(length(a)-1)) do identifiant
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
end
"""
    batched_sum(b::AbstractMatrix,nb_elements::AbstractVector)

compute the sum of a Concatenated batch with ndim  = 2. The first dim is the feature dimension. The second dim is the the batch dim.

Given `b` of size (n,m) and `nb_elements` of size (k,), the output has size (n,k).
"""
function batched_sum(b::AbstractMatrix, nb_elements::AbstractVector)
    a = similar(b, (size(b, 1), length(nb_elements) - 1))
    batched_sum!(a, b, nb_elements)
    a
end

function batched_sum(b::CuMatrix, nb_elements::AbstractVector)
    a = similar(b, eltype(b), (size(b, 1), length(nb_elements) - 1))
    batched_sum!(a, b, cu(nb_elements))
    a
end

function ChainRulesCore.rrule(::typeof(batched_sum), b::AbstractMatrix, nb_elements)
    res = batched_sum(b, nb_elements)
    function batched_sum_pullback(delta)::Tuple{NoTangent,Any,NoTangent}
        delta_b = @thunk begin
            delta_b = similar(b)
            Folds.foreach(minimum(eachindex(nb_elements)):(maximum(eachindex(nb_elements))-1)) do i
                delta_b[:, (nb_elements[i]+1):nb_elements[i+1]] .= delta[:, i]
            end
            delta_b
        end

        NoTangent(), delta_b, NoTangent()
    end
    res, batched_sum_pullback
end

function alloc_concatenated(sub_array, l)
    similar(
        sub_array,
        sub_array |> eltype,
        (size(sub_array)[begin:end-1]..., l))
end

function evaluate_and_cat(arrays, n::Int, sub_array, get_slice)
    indexes = 1:n
    res = alloc_concatenated(sub_array, get_slice(n) |> last)
    foreach(indexes) do i
        @inbounds view(res, fill(:, ndims(sub_array) - 1)..., get_slice(i)) .= arrays(i)
    end
    res
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(evaluate_and_cat), arrays, n::Int, sub_array, get_slice)
    indexes = 1:n
    res = alloc_concatenated(sub_array, get_slice(n) |> last)
    pullbacks = Array{Function}(undef, n)
    Folds.foreach(indexes) do i
        res[fill(:, ndims(sub_array) - 1)..., get_slice(i)], pullbacks[i] = rrule_via_ad(config, arrays, i)
    end
    function pullback_evaluate_and_cat(dres)
        map(indexes) do i
            pullbacks[i](dres[fill(:, ndims(sub_array) - 1)..., get_slice(i)])
        end, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    res, pullback_evaluate_and_cat
end
