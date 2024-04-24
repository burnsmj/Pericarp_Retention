#!/bin/bash
echo -e Color_Space'\t'Metric'\t'Linkage'\t'R_retained_coverage'\t'R_normret_coverage'\t'R_propret_coverage'\t'R_normprop_coverage'\t'R_retained_quantity'\t'R_normret_quantity'\t'R_propret_quantity'\t'R_normprop_quantity > Unsupervised_Annotation_Iterations.txt

for file in *.out; do 
	cs=$(head -1 $file)
	met=$(head -2 $file | tail -1)
	link=$(head -3 $file | tail -1)
	rrc=$(head -22 $file | tail -1 | rev | cut -d' ' -f 1 | rev)
	rnc=$(head -23 $file | tail -1 | rev | cut -d' ' -f 1 | rev)
	rpc=$(head -24 $file | tail -1 | rev | cut -d' ' -f 1 | rev)
	rnpc=$(head -25 $file | tail -1 | rev | cut -d' ' -f 1 | rev)
	rrq=$(head -26 $file | tail -1 | rev | cut -d' ' -f 1 | rev)
	rnq=$(head -27 $file | tail -1 | rev | cut -d' ' -f 1 | rev)
	rpq=$(head -28 $file | tail -1 | rev | cut -d' ' -f 1 | rev)
	rnpq=$(head -29 $file | tail -1 | rev | cut -d' ' -f 1 | rev)

	echo -e $cs'\t'$met'\t'$link'\t'$rrc'\t'$rnc'\t'$rpc'\t'$rnpc'\t'$rrq'\t'$rnq'\t'$rpq'\t'$rnpq >> Unsupervised_Annotation_Iterations.txt
done
