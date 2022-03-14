# PU-OC
This repository accompanies the paper "Improving State-of-the-Art in One-Class Classification by Leveraging Unlabeled Data".

* OC models with their PU modifications can be found in `models`

* Pretrained autoencoder weights can be found in `AE weights`

* OC and PU models tested in experiment 5.3, A.6, and A.5 can be found in `models/reference`

* Testing functions for all settings can be found in `test` 

    * Code for experiment **One-vs-all** can be found at `test/drocc/one-vs-all.py` for DROCC-base models and at `test/one-vs-all.py` for other 
    * Code for experiment **Number of positive modes** can be found at `test/drocc/pos modality.py` for DROCC-base models and at `test/pos modality.py` for other  
    * Code for experiment **Shift of the negative distribution** can be found at `test/drocc/neg shift.py` for DROCC-base models and at `test/neg shift.py` for other  
    * Code for experiment **Size and contamination of unlabeled data** can be found  at `test/unlabeled.py`
    * Code for **Abnormal1001** same as **One-vs-all**  
    * Code for lstm-based models can be found in `lstm.ipynb`
    