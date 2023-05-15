#!/bin/sh

#PBS -N compilationStage

#PBS -M andref.castellanos@cinvestav.mx

#PBS -q CGPUA100

#PBS -l walltime=0:15:00:nodes=1:ppn=12

#PBS -o mypath/my.out

#PBS -e mypath/my.err

#PBS -W stagein=file_list

#PBS -W stageout=file_list

#PBS -m abe

#PBS -r n

source /opt/gpu/cudavars.sh 11.0
