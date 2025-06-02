function evaluate_trivial(volume::AbstractArray{Float32,3},mins::Point3{Int},atoms::RegularGrid)
	cutoff_radius = atoms.radius
	unknow_indices = Point3{Int}[]
	for (i,j,k)  in eachindex(IndexCartesian(),volume)
		pos  = mins .+ cutoff_radius .* (i,j,k) |> Point3f
		has_atom_nearby = false
		__inrange(atoms,pos) do s::Sphere{Float32}
			d = (s.pos .- pos ) .^2 |> sum
			if d < cutoff_radius^2
				volume[i,j,k] = 1f0
			elseif d < (cutoff_radius + r)^2
				has_atoms_nearby = true
			end
		end
		if !has_atoms_nearby
			volume[i,j,k] = 0f0
		end
		if volume[i,j,k] not in (0f0,1f0)
			push!(unknown_indices,Point3(i,j,k))
		end
	end
	unknown_indices
end
function evaluate_field_fast(model::StatefulLuxLayer,atoms::RegularGrid;step::Number=1,batch_size = 100000)::Array{Float32,3}
	mins = atoms.start .- 2
	maxes = mins .+ size(atoms.grid) .* atoms.radius .+ 2
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1,1), reshape(ranges[2], 1, :,1), reshape(ranges[3], 1,1,:))
    volume = similar(grid,Float32)
    unknown_indices = evaluate_trivial!(volume,mins,atoms)
    for i in 1:batch_size:length(unknown_indices)
    	k = min(i+ batch_size-1,length(v))
    	v=  view(unknown_indices,i:k)
    	res =  reshape(model((Batch(mins .+ atoms.radius .* v), atoms)) |> cpu_device(),:)
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
