#include "MLNanoShaperRunner.h"
#include "julia_init.h"

int main(int argc,char *argv[]) {
  init_julia(argc, argv);
  if(load_model("tiny_angular_dense_s_jobs_11_6_3_c_2025-03-10_epoch_800_10631177997949843226") !=0)
    return -2;
  sphere data[2]= {{0.,0.,0.,1.},{1.,0.,0.,1.}};
  if(load_atoms(data,2) !=0)
    return -1;
  point x[2] = {{0.,0.,1.},{1.,0.,0.}};
  eval_model(x,2);
  shutdown_julia(0);

  return 0;
}
