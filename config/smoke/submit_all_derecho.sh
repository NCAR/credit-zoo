#!/bin/bash
# Peer-submit all model-zoo smoke tests to Derecho's gpudev queue (via desched1),
# routed through 'develop@desched1' since direct gpu/gpudev@desched1 submission
# from Casper is access-denied but the 'develop' front door routes there fine.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS=(
    smoke_stormer_casper.yml
    smoke_arches_casper.yml
    smoke_sfno_casper.yml
    smoke_climax_casper.yml
    smoke_fourcastnet_casper.yml
    smoke_fourcastnet3_casper.yml
    smoke_fengwu_casper.yml
    smoke_swinrnn_casper.yml
    smoke_itransformer_casper.yml
    smoke_mambavision_casper.yml
    smoke_graphcast_casper.yml
    smoke_healpix_casper.yml
    smoke_corrdiff_casper.yml
    smoke_fuxi_ens_casper.yml
    smoke_aurora_casper.yml
    smoke_pangu_casper.yml
    smoke_aifs_casper.yml
    smoke_stormer_noinit_casper.yml
    smoke_climax_noinit_casper.yml
    smoke_fourcastnet_noinit_casper.yml
    smoke_aurora_noinit_casper.yml
    smoke_fuxi_noinit_casper.yml
    smoke_stormer_pretrained_casper.yml
    smoke_climax_pretrained_casper.yml
    smoke_fourcastnet_pretrained_casper.yml
    smoke_aurora_pretrained_casper.yml
    smoke_fuxi_pretrained_casper.yml
)

for cfg in "${CONFIGS[@]}"; do
    CFG_PATH="${SCRIPT_DIR}/${cfg}"
    JOB_NAME="${cfg%.yml}"
    SAVE_LOC="$(python3 -c "import yaml, os; c=yaml.safe_load(open('${CFG_PATH}')); print(os.path.expandvars(c['save_loc']))")"
    mkdir -p "${SAVE_LOC}"
    cp "${CFG_PATH}" "${SAVE_LOC}/model.yml"

    PBS_SCRIPT=$(cat <<EOF
#!/bin/bash -l
#PBS -N ${JOB_NAME}
#PBS -l select=1:ncpus=8:ngpus=1:mem=128GB
#PBS -l walltime=00:30:00
#PBS -A NAML0001
#PBS -q develop@desched1
#PBS -j oe
#PBS -k eod
#PBS -o ${SAVE_LOC}/${JOB_NAME}.derecho.out

module --force purge
module load ncarenv/24.12 nvhpc cuda/12.3.2 cray-mpich conda
conda activate credit-main-derecho

pip install -e /glade/work/schreck/repos/credit-zoo --no-deps -q

echo "Host: \$(hostname)"
python -m credit.applications.train_gen2 -c ${SAVE_LOC}/model.yml
EOF
)

    LAUNCH_SH="${SAVE_LOC}/launch_derecho.sh"
    echo "${PBS_SCRIPT}" > "${LAUNCH_SH}"
    JOBID=$(qsub "${LAUNCH_SH}" 2>&1)
    echo "Submitted ${cfg} -> ${JOBID}"
done
