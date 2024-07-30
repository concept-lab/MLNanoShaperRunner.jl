#include "MLNanoShaperRunner.h"
#include "julia_init.h"

int main(int argc,char *argv[]) {
  init_julia(argc, argv);
  return 0;
  load_model("/home/tristan/datasets/models/"
               "angular_dense_2Apf_epoch_10_16451353003083222301");
  sphere data[2]= {{0.,0.,0.,1.},{1.,0.,0.,1.}};
  load_atoms(data,2);
  point x[2] = {{0.,0.,1.},{1.,0.,0.}};
  eval_model(x,2);
  shutdown_julia(0);

  return 0;
}
