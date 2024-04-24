#!/bin/bash
for cs in RGB LAB HSV XYZ LUV YIQ YUV; do
	for metric in CHI SC DBI; do
		for linkage in ward single average complete; do
			sbatch --export=CS=$cs,METRIC=$metric,LINKAGE=$linkage unsupervised_annotation.sh
		done
	done
done
