# Configuration Files

Each tool has a specific type of configuration file format. Several of the bash files here reflect different configurations for the same tool, to do different things. Below is the description of each config format, and the current bash files using them as well.

## Trainer Tool

The Trainer tool configuration has the following fields:

 - **model_module**: name of the Python module containing the model-specific functions to be used when training (see main README for details). Since it has to be inside a folder inside the `harness/datasets` subfolder, it needs to has the format "<sub-folder>.<model_module_name>" (i.e., "iceberg.iceberg_model"). 
- **dataset_class**: name of the dataset class extending DataSet and implementing dataset-specific functions (see main README for details). Since it has to be inside a folder inside the `harness/datasets` subfolder, it has the format "<sub-folder>.<dataset_module_name>.<class_name>" (i.e.,  "iceberg.iceberg_dataset.IcebergDataSet")
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

 - **mode**: "drift" to generated drifted dataset, "test" to test an already generated one.
 - **dataset_class**: same as config field in Trainer tool.
 - **dataset**: relative path to a JSON file with the labelled dataset to use for generating a drifted one.
 - **output**: relative path to a JSON file that will contain the drifted dataset, referencing the **dataset** above by containing only new ids and original_ids pointing to samples in it. In test mode, this is the tested dataset.
 - **bins**: array defining bins to split the **dataset** into when generating drift. Each item in the array will also be an array, containing first the bin name, and then the bin order (i.e, ["no_iceberg", 0])
 - **drift_scenario**: information about the drift being generated.
   - **condition**: friendly name for the drift type. 
   - **module**: Python module implementing the drift (see general README for more details), which has to go inside the `harness/drift/` (i.e., "random_drift").
   - **params**: dictionary of specific parameters for this drift generator. Depend on the **module** defined above. It will at least contain the following minimum parameters:
     - **max_num_samples**: how many samples to output into the drifted dataset.
     - **timebox_size**: size of the timebox for generating the drifted dataset. 

The bash scripts for this tool include:
- **drifter.sh**: uses the `drifter_config.json` file, default for creating a drifted dataset.

## Predictor Tool

The Predictor tool configuration has the following fields:

 - **dataset_class**: same as config field in Trainer tool.
 - **dataset**: relative path to a JSON file that will contain the dataset to predict on (which can be a full dataset or a reference dataset).
 - **base_dataset**: relative path to a JSON file for a dataset. If present, it means **dataset** is a reference dataset, and this is the base it is referencing. If abset, **dataset** is a full dataset.
 - **model**: relative path to the folder where the trained Tensorflow SaveModel to be used is stored.
 - **mode**: can be "predict", which will execute the model and output predictions and metrics, or "label", which will execute the model but write an updated labelled dataset with that output.
 - **output**: relative path to JSON file where the predictions will be stored.
 - **metrics_output**: relative path to the JSON file where the metric information will be stored.
 - **threshold**: value between 0 and 1 that will convert a raw prediction into a classification.
 - **timebox_size**: size of the timebox for analyzing the metrics, should match timebox used when creating a drifted dataset. Only needed if "predict" mode is enabled.
 - **metrics**: array containing objects describing the metrics to analyze. Only needed if "predict" mode is enabled. Each metric object contains:
   - **name**: a friendly name for the metric.
   - **type**: the metric type (can only be ErrorMetric or DistanceMetric).
   - **module**: name of the Python module inside the `harness/drifts` folder implementing this drift (see general README for more details).
   - **params**: dictionary containing drift module specific parameters. For DistanceType metrics, this can contain a **pdf_params** parameter which itself contains PDF-specific parameters.

The bash scripts for this tool include:
- **predictor.sh**: uses the `predictor_config.json` file, default for predicting and generating metrics.
- **labeller.sh**: uses the `labeller_config.json` file, uses the "label" mode to create an updated dataset with the predictions as its labels.

## Merger Tool

This is a separate, simpler tool. Its configuration has the following fields:

 - **dataset1**: path to first JSON dataset to merge.
 - **dataset2**: path to second JSON dataset to merge.
 - **output**: path to JSON file with merged dataset.

The bash scripts for this tool include:
- **merger.sh**: uses the `merger_config.json` file, default for merging two datasets.
