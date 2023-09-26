#!/bin/bash
#SBATCH -p q1
#SBATCH --job-name SVDOMP
#SBATCH -n 36
#SBATCH -o salida.%j.txt
#SBATCH -e err-%j.log
#SBATCH --output=SVDOMP-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andref.castellanos@cinvestav.mx

export OMP_NUM_THREADS=36

module load numactl-2.0.14-gcc-10.2.0-62op4tg hwloc-2.2.0-gcc-10.2.0-it4o6fq glib-2.66.2-gcc-10.2.0-wcwvaat gcc-10.2.0-gcc-10.2.0-u6fkk3y openmpi/4.0.3/gcc cmake-3.18.4-gcc-10.2.0-ch2vvyp

cmake ..
make
#mpiexec --map-by ppr:1:node --bind-to core -x OMP_NUM_THREADS -v -display-map -display-allocation SVD_Jacobi_MPI_OMP 1000
mpiexec --bind-to hwthread --map-by ppr:1:node:pe=36 -x OMP_NUM_THREADS -v -display-map -display-allocation SVD_Jacobi_MPI_OMP 1000
mpiexec --bind-to hwthread --map-by ppr:1:node:pe=36 -x OMP_NUM_THREADS -v -display-map -display-allocation SVD_Jacobi_MPI_OMP 2000
mpiexec --bind-to hwthread --map-by ppr:1:node:pe=36 -x OMP_NUM_THREADS -v -display-map -display-allocation SVD_Jacobi_MPI_OMP 5000
mpiexec --bind-to hwthread --map-by ppr:1:node:pe=36 -x OMP_NUM_THREADS -v -display-map -display-allocation SVD_Jacobi_MPI_OMP 7000
mpiexec --bind-to hwthread --map-by ppr:1:node:pe=36 -x OMP_NUM_THREADS -v -display-map -display-allocation SVD_Jacobi_MPI_OMP 10000
mpiexec --bind-to hwthread --map-by ppr:1:node:pe=36 -x OMP_NUM_THREADS -v -display-map -display-allocation SVD_Jacobi_MPI_OMP 20000
mpiexec --bind-to hwthread --map-by ppr:1:node:pe=36 -x OMP_NUM_THREADS -v -display-map -display-allocation SVD_Jacobi_MPI_OMP 30000
