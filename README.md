## Implementation for COMP0138 Individual Project - Candidate YSJP3
This repository contains code that was used to conduct the experiments in the individual project (COMP0138), as part of the completion of MEng Computer Science programme at UCL.

It contains code that were modified from the original repository [Topological_Feature_Selection](https://github.com/FinancialComputingUCL/Topological_Feature_Selection) implemented by [A. Briola](https://github.com/AntoBr96). 

## Usage guide 
### 1. Construct the similarity matrix for a specific dataset. 
Run ```python tmfg_fs_out.py --stage SM_COMPUTATION --dataset <dataset_name> --cc_type <similarity_measure>```

e.g. ```python tmfg_fs_out.py --stage SM_Computation --dataset lung_small --cc_type pearson``` to create a similarity matrix using the Pearson's correlation coefficient for the "lung_small" dataset. 

### 2. Perform Topological Feature Selection + Training stage for a specific configuration. 
Run ```python tmfg_fs_out.py --stage TMFG_FS --dataset <dataset_name> --centrality <centrality_measure> --corr_type <similarity_value_type> --unweighted <weightedness_boolean> --edge_type <weighted_edge_type> --classification_algo KNN```

e.g. ```python tmfg_fs_out.py --stage TMFG_FS --dataset lung_small --centrality degree --corr_type square --unweighted false --edge_type sq --classification_algo KNN``` to perform TFS and train the KNN classifier using the weighted TMFG constructed from squared similarity values, with features selected based on degree centrality calculated with square edge weights.  

### 3. Perform TFS + Testing stage for a specific configuration. 
e.g. ```python tmfg_fs_out.py --stage TMFG_FS_TEST --dataset lung_small --centrality degree --corr_type square --unweighted false --edge_type sq --classification_algo KNN``` to perform out-of-sample tests on the configuration stated above. 

### 4. Perform subsampling for a specific configuration. 
Run ```python tmfg_fs_out.py --stage TMFG_FS_TEST --do_subsampling --dataset <dataset_name> --centrality <centrality_measure> --corr_type <similarity_value_type> --unweighted <weightedness_boolean> --edge_type <weighted_edge_type> --classification_algo <classifier_algorithm>```

# License

Copyright 2023 Antonio Briola, Tomaso Aste.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

```http://www.apache.org/licenses/LICENSE-2.0```

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
