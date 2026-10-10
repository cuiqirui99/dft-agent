#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
set -e
module load vasp
cat /home/researcher/potcar/potpaw_PBE/Fe/POTCAR > POTCAR
test -s POTCAR
sha256sum POSCAR INCAR KPOINTS > input_hashes.sha256
sha256sum POTCAR > potcar_hash.sha256
grep TITEL POTCAR > potcar_titles.txt
python3 - <<'PY'
import json,os,datetime
json.dump({"job_id":os.environ.get("SLURM_JOB_ID"),"started_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"tasks":os.environ.get("SLURM_NTASKS")},open("execution.json","w"))
PY
set +e
srun vasp_std
code=$?
set -e
python3 - "$code" <<'PY'
import json,sys,datetime
p=json.load(open("execution.json"));p.update(exit_code=int(sys.argv[1]),finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
json.dump(p,open("execution.json","w"))
PY
exit "$code"
