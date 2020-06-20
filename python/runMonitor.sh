#!/bin/zsh -l
#PBS -m be
#PBS -M jklymak@gmail.com
#PBS -l select=1:ncpus=1
#PBS -l walltime=3:10:30
#PBS -q transfer
#PBS -A ONRDC35552400
#PBS -j oe
#PBS -N plotIt

cd $PBS_O_WORKDIR
source /u/home/jklymak/.zshrc
echo ${PATH}

for todo in  AHMany0250_020amp400f10aw050cw050N0010 \
             AHMany0250_010amp400f10aw050cw050N0010 \
             AHMany0250_005amp400f10aw050cw050N0010
do
  /p/home/jklymak/miniconda3/bin/python plotBottom.py ${todo}
done
