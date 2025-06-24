function evaluate_trivial!(volume::AbstractArray{Float32,3},coordinates::AbstractArray{Point3f,3},atoms::RegularGrid)
	cutoff_radius = atoms.radius
	local_unknow_indices = [Vector{Point3{Int}}() for _ in 1:Threads.nthreads()]
    @threads for I in eachindex(IndexCartesian(), coordinates)
    	thread_id = Threads.threadid()
		i = Tuple(I)
		pos = coordinates[I]
		has_atoms_nearby = Ref(false)
		cutoff_radius² = cutoff_radius^2
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
			push!(local_unknow_indices[thread_id], Point3i(i...))
		end
	end
	unknow_indices = Point3{Int}[]
	for thread_indices in local_unknow_indices
        append!(unknow_indices, thread_indices)
    end
	unknow_indices
end
function coordinates(r::RegularGrid,i)
	r.start .+ r.radius .* i
end
function evaluate_field_fast(model::StatefulLuxLayer,atoms::RegularGrid;step::Number=1f0,batch_size = 100000)::Array{Float32,3}
	mins = atoms.start .- 2
	maxes = mins .+ size(atoms.grid) .* atoms.radius .+ 2
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1,1), reshape(ranges[2], 1, :,1), reshape(ranges[3], 1,1,:))
    volume = similar(grid,Float32)
    unknown_indices = evaluate_trivial!(volume,grid,atoms)
    return volume
    for i in 1:batch_size:length(unknown_indices)
    	k = min(i+ batch_size-1,length(unknown_indices))
    	v=  view(unknown_indices,i:k)
    	res =  reshape(model((Batch(map(v) do v coordinates(atoms,v) end), atoms)) |> cpu_device(),:)
    	for ((i,j,k),r) in zip(v,res)
	    	volume[i,j,k] = r 
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
