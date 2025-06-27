function count_point_types(coordinates::AbstractArray{Point3f,3},atoms::RegularGrid)
	cutoff_radius = atoms.radius
	cutoff_radius² = cutoff_radius^2
	local_unknown_counts=zeros(Int,Threads.nthreads())
	local_known_counts=zeros(Int,Threads.nthreads())
	local_outside_counts=zeros(Int,Threads.nthreads())
	@info length(coordinates)
    @threads for I in eachindex(IndexCartesian(), coordinates)
    	thread_id = Threads.threadid()
		i = Tuple(I)
		pos = coordinates[I]
		has_atoms_nearby = Ref(false)
		is_inside_atom = Ref(false)
		_iter_grid(atoms,pos,Δ3) do s::Sphere{Float32}
			d² = (s.center.- pos ) .^2 |> sum
			if d² < s.r^2
				is_inside_atom[] = true
				return true
			elseif d² < cutoff_radius²
				has_atoms_nearby[] = true
			end
			return false
		end
		if is_inside_atom[]
			local_known_counts[thread_id] +=1
		elseif has_atoms_nearby[]
			# @info "unknown indices" I
			local_unknown_counts[thread_id]+=1
		else
			local_outside_counts[thread_id]+=1
		end
	end
	(;known_counts = sum(local_known_counts),unknown_counts = sum(local_unknown_counts),outside_counts = sum(local_outside_counts))
end
function evaluate_trivial!(volume::AbstractArray{Float32,3},coordinates::AbstractArray{Point3f,3},atoms::RegularGrid)::Tuple{Vector{CartesianIndex{3}},Vector{Point3f}}
	cutoff_radius = atoms.radius
	cutoff_radius² = cutoff_radius^2
	local_unknow_indices = [Vector{CartesianIndex{3}}() for _ in 1:Threads.nthreads()]
	local_unknow_pos = [Vector{Point3f}() for _ in 1:Threads.nthreads()]
    for I in eachindex(IndexCartesian(), coordinates)
    	thread_id = Threads.threadid()
		i = Tuple(I)
		pos = coordinates[I]
		has_atoms_nearby = Ref(false)
		volume[I] = 0f0
		_iter_grid(atoms,pos,Δ3) do s::Sphere{Float32}
			d² = (s.center.- pos ) .^2 |> sum
			if d² < s.r^2
				volume[I] = 1f0
				has_atoms_nearby[] = false
				return true
			elseif d² < cutoff_radius²
				has_atoms_nearby[] = true
			end
			return false
		end
		if has_atoms_nearby[]
			# @info "unknown indices" I
			push!(local_unknow_indices[thread_id], I)
			push!(local_unknow_pos[thread_id], coordinates[I])
		end
	end
	unknow_indices = Vector{CartesianIndex{3}}()
	unknow_pos = Vector{Point3f}()
	for (thread_indices,thread_pos) in zip(local_unknow_indices,local_unknow_pos)
        append!(unknow_indices, thread_indices)
        append!(unknow_pos, thread_pos)
    end
	unknow_indices,unknow_pos
end
function evaluate_field_fast(model,atoms::RegularGrid;step::Number=1f0,batch_size = 100000)::Array{Float32,3}
	mins = atoms.start .- 2
	maxes = mins .+ size(atoms.grid) .* atoms.radius .+ 2
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1,1), reshape(ranges[2], 1, :,1), reshape(ranges[3], 1,1,:))
    volume = similar(grid,Float32)
    unknown_indices,unknown_pos = evaluate_trivial!(volume,grid,atoms)
    for i in 1:batch_size:length(unknown_indices)
    	k = min(i+ batch_size-1,length(unknown_indices))
    	v=  view(unknown_indices,i:k)
    	p=  view(unknown_pos,i:k)
    	res::Vector{Float32} =  vec(model((Batch(p), atoms))::Array{Float32})
    	@assert length(res) == length(v)
    	for (I,r) in zip(v,res)
	    	volume[I] = r 
    	end
    end
	volume
end
function evaluate_field(model::StatefulLuxLayer,atoms::RegularGrid;step::Number=1,batch_size = 100000)::Array{Float32,3}
	mins = atoms.start .- 2
	maxes = mins .+ size(atoms.grid) .* atoms.radius .+ 2
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1,1), reshape(ranges[2], 1, :,1), reshape(ranges[3], 1,1,:))
    g = reshape(grid,:)
    volume = similar(grid,Float32)
    v = reshape(volume,:)
    for i in 1:batch_size:length(volume)
    	k = min(i+ batch_size-1,length(v))
    	res =  reshape(model((Batch(view(g,i:k)), atoms)) |> cpu_device(),:)
    	v[i:k] .= res
        end
	volume
end
