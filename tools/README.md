# Configuration Files

Each tool has a specific type of configuration file format. Several of the bash files here reflect different configurations for the same tool, to do different things. Below is the description of each config format, and the current bash files using them as well.

## Trainer Tool

The Trainer tool configuration has the following fields:

 - **model_module**: name of the Python module containing the model-specific functions to be used when training (see main README for details). Since it has to be inside a folder inside the `harness/datasets` subfolder, it needs to has the format "<sub-folder>.<model_module_name>" (i.e., "iceberg.iceberg_model"). 
 - **dataset_class**: name of the dataset class extending DataSet and implementing dataset-specific functions (see main README for details). Similar to **model_module**, it has the format "<sub-folder>.<dataset_module_name>.<class_name>" (i.e.,  "iceberg.iceberg_dataset.IcebergDataSet")
 - **dataset**: relative path to a JSON file with the labelled dataset to use for training.
 - **model**: relative path to the folder where the trained Tensorflow SaveModel model will be stored (or is already stored, for the evaluation mode).
 - **training**: only if "on", it will train a new model using **dataset** and store it in **model**.
 - **cross_validation**: if "on", it will perform k-fold cross validation with the given **dataset**, to compare the results of training with different subsets of the data. This will just present results in the log, and won't store a new model.
 - **evaluation**: if "on", this will load the model indicated in **model** and evaluate it, using the full **dataset** if **training** was off, or the data split for validation if **training** was on (so it was executed).
 - **hyper_parameters**: parameters used when training the model. Only used if **training** is "on".
   - **epochs**: number of epochs to use while training.
   - **batch_size**: size of each batch to use while training.
   - **validation_size**: % of the dataset to split for validation.

The bash scripts for this tool include:
 - **trainer.sh**: uses the `trainer_config.json` file, default for training, with no evaluation and no cross-validation.
 - **eval.sh**: uses the `eval_config.json` file, has cross-validation and evaluation on, but no model training on.

## Drifter Tool

The Drifter tool configuration has the following fields:

 - **dataset_class**: name of the dataset class extending DataSet and implementing dataset-specific functions (see main README for details). Since it has to be inside a folder inside the `harness/datasets` subfolder, it has the format "<sub-folder>.<dataset_module_name>.<class_name>" (i.e.,  "iceberg.iceberg_dataset.IcebergDataSet")
 - **dataset**: relative path to a JSON file with the labelled dataset to use for generating a drifted one.
 - **output**: relative path to a JSON file that will contain the drifted dataset, referencing the **dataset** above by containing only new ids and original_ids pointing to samples in it.
 - **bins**: array defining bins to split the **dataset** into when generating drift. Each item in the array will also be an array, containing first the bin name, and then the bin order (i.e, ["no_iceberg", 0])
 - **drift_scenario**: information about the drift being generated.
   - **condition**: friendly name for the drift type. 
   - **module**: Python module implementing the drift (see general README for more details), which has to go inside the `harness/drift/` (i.e., "random_drift").
   - **params**: dictionary of specific parameters for this drift generator. Depend on the **module** defined above.

The bash scripts for this tool include:
- **drifter.sh**: uses the `drifter_config.json` file, default for creating a drifted dataset.
