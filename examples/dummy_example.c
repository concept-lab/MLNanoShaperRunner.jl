#include "MLNanoShaperRunner.h"
#include "julia_init.h"

int main(int argc,char *argv[]) {
  init_julia(argc, argv);
  return 0;
  load_model("tiny_angular_dense_3.0A_smooth_14_categorical_2024-08-02_epoch_70_18127875713564776610");
  sphere data[2]= {{0.,0.,0.,1.},{1.,0.,0.,1.}};
  load_atoms(data,2);
  point x[2] = {{0.,0.,1.},{1.,0.,0.}};
  eval_model(x,2);
  shutdown_julia(0);

  return 0;
}
