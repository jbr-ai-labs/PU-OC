# PU-OC
This repository accompanies the paper "Improving State-of-the-Art in One-Class Classification by Leveraging Unlabeled Data".

# Structure

* OC models with their PU modifications can be found in `models`

* Pretrained autoencoder weights can be found in `AE weights`

* OC and PU models tested in experimentы 5.3, A.6, and A.5 can be found in `models/reference`

* Testing functions for all settings can be found in `test` 

    * Code for the experiment **One-vs-all** can be found in `test/drocc/one-vs-all.py` for DROCC-based models and in `test/one-vs-all.py` for other models
    * Code for the experiment **Number of positive modes** can be found in `test/drocc/pos modality.py` for DROCC-based models and in `test/pos modality.py` for other models
    * Code for the experiment **Shift of the negative distribution** can be found in `test/drocc/neg shift.py` for DROCC-based models and in `test/neg shift.py` for other models
    * Code for the experiment **Size and contamination of unlabeled data** can be found in `test/unlabeled.py`
    * Code for **Abnormal1001** is the same as for **One-vs-all**  
    * Code for the lstm-based models can be found in `lstm.ipynb`
    
