#!/bin/bash

# run as: source runAll.sh LW1kmlowU10Amp305K18 10 10 100

for todo in AHMany0250_020amp400f10aw050cw050N0010 \
            AHMany0250_010amp400f10aw050cw050N0010 \
            AHMany0250_005amp400f10aw050cw050N0010
do
    echo ${todo}
    mainr=$(qsub -N ${todo} runModel.sh)
    TransferRun=$(qsub -W depend=afterany:$mainr -N ${todo} transfertoarchive.sh)
done
