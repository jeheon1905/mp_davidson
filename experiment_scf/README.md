# SCF experiments

```bash
# etol=1e-6, dtol=1e-4
source test.scf.sh ./experiment_scf/config.CNT_6_0.sh
source test.scf.sh ./experiment_scf/config.Si_diamond.sh
source test.scf.sh ./experiment_scf/config.MgO.sh

# etol=1e-7, dtol=1e-5 (with lower convergence criteria)
source test.scf.2.sh ./experiment_scf/config.CNT_6_0.sh
source test.scf.2.sh ./experiment_scf/config.Si_diamond.sh
source test.scf.2.sh ./experiment_scf/config.MgO.sh
```
