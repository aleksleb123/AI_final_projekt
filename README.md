Navodila uporabe:
  1. Downlodaj vse
  2. Jaz uporabljam PyCharm, kot vmesnik za python in imam vse v enem direktoriju izgleda kot je na sliki.<img width="666" height="551" alt="Izgled direktorija" src="https://github.com/user-attachments/assets/29d42d1c-3beb-4113-8b60-4ad5eaaa4fce" />
  3. Preimenuj "_.vscode" v ".vscode"
  4. Downlodaj iz "Images_FDG_PET" in "Images_MRI_T1" iz spletne učilnice https://labkey-public.fmf.uni-lj.si/labkey/ADNI/project-begin.view
  5. Vse kar naredijo python files se bo shranil v folder, ki se bo sam naredil, "saved_models"

Moj vrstni red poganjanja datotek:
  0. Nastavitve se spreminja v config.toml. POZOR!! model se trenira DOLGO ČASA, jaz sem treniral 5 modelov 20 epoh 2-3 dni. 
  1. Train_cnn
  2. ensemble_predict
  3. threshold_xarray
  4. bayesian
  5. sensitivity_analysis_fixed.py, (sensitivity_analysis, je stara verzija)
  6. xarray_sensitivity.py
  7. xarray_images.py
  8. threshold_refac.py
  9. threshold.py


