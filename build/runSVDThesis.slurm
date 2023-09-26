#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodelist nvd01
#SBATCH --job-name SVDHet
#SBATCH -n 16
#SBATCH -o salida.%j.txt
#SBATCH -e err-%j.log
#SBATCH --output=SVDThesis-%j.out

export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

module load cmake-3.18.4-gcc-10.2.0-ch2vvyp

cmake ..
make
./SVD_Thesis