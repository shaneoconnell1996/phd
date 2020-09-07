#!/bin/sh
#SBATCH -p highmem
#SBATCH -J "fsl-cases"
#SBATCH --out="./cases.out"
#SBATCH -n 1
source /home/soconnell/UK_biobank_pipeline_v_1-master/init_vars
f=$1
echo "Processing $f"
FNIRT_REF=$MNI
FLIRT_REF=$MNI_brain
BASEDIR='/home/soconnell'
filename=$(basename "$f")
outname="${filename%.*.*.*}"
bet $f $BASEDIR/bet_struc/$outname.betted.nii.gz -R 
flirt -ref $FLIRT_REF -in $BASEDIR/bet_struc/$outname.betted.nii.gz -omat $BASEDIR/new_warp_coefs/$outname.affine.mat
convert_xfm -omat $BASEDIR/new_warp_coefs/$outname_T1_to_orig.mat -inverse $BASEDIR/new_warp_coefs/$outname.affine.mat
convert_xfm -omat $BASEDIR/new_warp_coefs/$outname.linear.mat -concat $BASEDIR/new_warp_coefs/$outname_T1_to_orig.mat $BASEDIR/new_warp_coefs/$outname.affine.mat

fnirt --in=$f --aff=$BASEDIR/new_warp_coefs/$outname.affine.mat --cout=$BASEDIR/new_warp_coefs/$outname.warpcoef.nii.gz --config=T1_2_MNI152_2mm --interp=spline
applywarp --ref=$FNIRT_REF --in=$f --warp=$BASEDIR/new_warp_coefs/$outname.warpcoef.nii.gz -o $BASEDIR/processed_cases/$outname.registered.adjusted_bbconf.nii.gz --interp=spline
echo "Processed $outname and wrote to directory"
