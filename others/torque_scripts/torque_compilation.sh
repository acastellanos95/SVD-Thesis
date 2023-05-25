#!/bin/sh

#PBS -N compilationStage

#PBS -M andref.castellanos@cinvestav.mx

#PBS -q CGPUA100

#PBS -l walltime=0:15:00:nodes=1:ppn=12

#PBS -o /home/ameneses/my.out

#PBS -e /home/ameneses/my.err

#PBS -m abe

#PBS -r n

source /opt/gpu/cudavars.sh 11.0

cd SVD-Thesis
cd build
cmake ..
make
cd ../..
