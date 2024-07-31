typedef struct {
  float x;
  float y;
  float z;
  float r;
} sphere;

typedef struct {
  float x;
  float y;
  float z;
} point;
// void shutdown_julia(int retcode);

/*
 *Load the model `parameters` form a serialised training state at absolute path
 *`path`. Parameters: path - the path to a serialized NamedTyple containing the
 *parameters of the model Return value(int):
 *- 0: OK
 *- 1: file not found
 *- 2: file could not be deserialized properly
 *- 3: unknow error
 */

int load_model(char *path);
/*
    load_atoms(start::Ptr{CSphere},length::Cint)::Cint

Load the atoms into the julia model.
Start is a pointer to the start of the array of `CSphere` and `length` is the
length of the array

# Return an error status:
- 0: OK
- 1: data could not be read
- 2: unknow error
*/
int load_atoms(sphere *start, int length);
/*
 *evaluate the model at coordinates start[0],...,start[length -1]
 */
float eval_model(point *start,int length); 
