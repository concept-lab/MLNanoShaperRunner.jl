using Meshing
"""
    generate_surface

generate the surface from a given model
"""
function generate_surface(atoms,training_states) 
    (; mins, maxes) = atoms.hyper_rec
	surface = isosurface(;origin=mins,width=maxes-mins)do x
        atoms_neighboord = atoms[inrange(atoms_tree, x, r)] |> StructVector
		if length(atoms_neighboord) >= 0
			training_states.model(atoms_neighboord,training_states.parameters,training_states.states) |> first
		else
			0.f0
		end
	end
end
