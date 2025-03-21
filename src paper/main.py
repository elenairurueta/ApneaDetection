# we used channel C4-A1
# divided the data into 30-second segments
# labelled all segments that contained al least 10 continuous seconds of OSA, MA or hypopnea as apnea
# all other segments were labelled as non-apnea

# PREPROCESSING
# 1. Lowpass filter (125 Hz)
# 2. Downsample
# 3. Z-score normalization
# 4. Segmentation and labeling
# 5. Undersampling
# 10-fold cross-validation
# Network training was performed for 40 epochs
# Cross-entropy loss function. 
# optimized using Adam: alpha coefficient was set at 0.9, and beta coefficient at 0.999. 
# The learning rate was tuned alongside other hyperparameters. lr = 0.00163

# To quantify performance of our trained CNN, we used accuracy and Matthews correlation coefficient (MCC).
# We used a Bayesian t-test, computing 95% highest density intervals (HDIs), to compare our CNN’s performance to baseline.