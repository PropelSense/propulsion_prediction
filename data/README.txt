
This file explains the structure of the Shift Benchmark on Marine Cargo Vessel Power Consumption prediction - see https://arxiv.org/pdf/2206.15407.pdf . 
This data is shared under a public CC BY NC SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/) .

----------------------------------------------------------------
-------------------- DATASET DESCRIPTION -----------------------
----------------------------------------------------------------

The data is split into 'real' and 'synthetic' data, with corresponding directories. Within each directory, there is:
 - training data
 - in-domain development data
 - shifted development data
 - in-domain eval data
 - shifted eval data

Additionally, the synthetic data also contains a 'generalisation set', which samples from the convex hull of possible inputs.
All data is contains in csv format. Thus, the structure of the full dataset is the following:

--real
  --train.csv
  --dev_in.csv
  --dev_out.csv
  --eval_in.csv
  --eval_out.csv

--synthetic
  --train.csv
  --dev_in.csv
  --dev_out.csv
  --eval_in.csv
  --eval_out.csv
  --generalisation_set.csv

Note, however, the initially only the synthetic train and dev sets will be released for Phase 1 of the Shifts Challenge 2.0 .
The 'real' training and dev sets will be released for Phase 2 of the Shifts Challenge 2.0, which begins March 1 2023.
The evaluation and generalisation sets will be kept hidden and will only be available for evaluation via Grand Challenge.