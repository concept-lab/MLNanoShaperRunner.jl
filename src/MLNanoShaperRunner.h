typedef struct {
	float x;
	float y;
	float z;
	float r;
} Sphere;

// void init_julia(int argc, char *argv[]);
// void shutdown_julia(int retcode);

/*
 *Load the model `parameters` form a serialised training state at absolute path `path`.
 *Parameters:
 *	path - the path to a serialized NamedTyple containing the parameters of the model
 *Return value(int):
 *- 0: OK
 *- 1: file not found
 *- 2: file could not be deserialized properly
 *- 3: unknow error
 */

int load_model(char* path);
/*
    load_atoms(start::Ptr{CSphere},length::Cint)::Cint

Load the atoms into the julia model.
Start is a pointer to the start of the array of `CSphere` and `length` is the length of the array

# Return an error status:
- 0: OK
- 1: data could not be read
- 2: unknow error
*/
int load_atoms(Sphere* start,int length);
/*
    eval_model(x::Float32,y::Float32,z::Float32)::Float32

evaluate the model at coordinates `x` `y` `z`.
*/
float eval_model(float x,float y,float z);
