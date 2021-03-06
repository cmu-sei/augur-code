# Configuration Files

Each tool has a specific type of configuration file format. Several of the bash files here reflect different configurations for the same tool, to do different things. Below is the description of each config format, and the current bash files using them as well.

## Trainer Tool

The Trainer tool configuration has the following fields:

 - **model_module**: name of the Python module containing the model-specific functions to be used when training (see main README for details). Since it has to be inside a folder inside the `harness/datasets` subfolder, it needs to has the format "<sub-folder>.<model_module_name>" (i.e., "iceberg.iceberg_model"). 
- **dataset_class**: name of the dataset class extending DataSet and implementing dataset-specific functions (see main README for details). Since it has to be inside a folder inside the `harness/datasets` subfolder, it has the format "<sub-folder>.<dataset_module_name>.<class_name>" (i.e.,  "iceberg.iceberg_dataset.IcebergDataSet")
 - **dataset**: relative path to a JSON file with the labelled dataset to use for training.
 - **model**: relative path to the folder where the trained Tensorflow SaveModel model will be stored (or is already stored, for the evaluation mode).
  - **ts_model**: relative path to the folder where the trained time-series model will be stored.
 - **training**: only if "on", it will train a new model using **dataset** and store it in **model**.
 - **cross_validation**: if "on", it will perform k-fold cross validation with the given **dataset**, to compare the results of training with different subsets of the data. This will just present results in the log, and won't store a new model.
 - **evaluation**: if "on", this will load the model indicated in **model** and evaluate it, using the full **dataset** if **training** was off, or the data split for validation if **training** was on (so it was executed).
 - **time_series_training**: only if "on", it will train the time-series model using **dataset** and store it in **ts_model**.
 - **hyper_parameters**: parameters used when training the model. Only used if **training** is "on".
   - **epochs**: number of epochs to use while training.
   - **batch_size**: size of each batch to use while training.
   - **validation_size**: % of the dataset to split for validation.
 - **time_interval**: time interval configuration. Only used if **time_series_training** is "on".
   - **starting_interval**: time interval at which start aggregating the data for creating the time-series and training its model.
   - **interval_unit**: time interval unit. Possible values available here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
   - **samples_per_time_interval**:  (OPTIONAL) only needed if dataset does not have timestamp data. In this case, this is the number of data samples that will be aggregated by time interval (instead of using timestamps for this aggregation).
 - **ts_hyper_parameters**: parameters used when training the time-series model. Only used if **time_series_training** is "on".
   - **order_p, order_q, order_d**: ARIMA training parameters.

The bash scripts for this tool include:
 - **trainer.sh**: uses the `trainer_config.json` file, default for training, with no evaluation and no cross-validation.
 - **eval.sh**: uses the `eval_config.json` file, has cross-validation and evaluation on, but no model training on.

## Drifter Tool

The Drifter tool configuration has the following fields:

 - **dataset_class**: same as config field in Trainer tool.
 - **dataset**: relative path to a JSON file with the labelled dataset to use for generating a drifted one.
 - **output**: relative path to a JSON file that will contain the drifted dataset, referencing the **dataset** above by containing only new ids and original_ids pointing to samples in it. In test mode, this is the tested dataset.
 - **bin_shuffle**: (OPTIONAL) true or false to indicate whether to shuffle samples in each bin after sorting them. Defaults to true.
 - **bin_values**: (OPTIONAL) dataset values to use when sorting into bins. Current values: "all" (one bin with everything), "results" for results/output/truth values. Defaults to "results".
 - **bins**: array defining bins to split the **dataset** into when generating drift. Each item in the array will also be an array, containing first the bin name, and then the bin order (i.e, ["no_iceberg", 0])
 - **timestamps**: timestamp generation parameters for the drifted dataset.
   - **start_datetime**: timestamp for the first element of the drifted dataset.
   - **increment_unit**: increment unit for each new sample in the drifted dataset. Possible values available here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
 - **drift_scenario**: information about the drift being generated.
   - **condition**: friendly name for the drift type. 
   - **module**: Python module implementing the drift (see general README for more details), which has to go inside the `harness/drift/` (i.e., "random_drift").
   - **params**: dictionary of specific parameters for this drift generator. Depend on the **module** defined above. It will at least contain the following minimum parameters:
     - **max_num_samples**: how many samples to output into the drifted dataset.
     - **sample_group_size**: size of the sample group for generating the drifted dataset.
     - **sample_group_shuffle**: (OPTIONAL) true or false to indicate whether to shuffle samples in each group after selecting them from bins. Defaults to true.

The bash scripts for this tool include:
- **drifter.sh**: uses the `drifter_config.json` file, default for creating a drifted dataset.

## Predictor Tool

The Predictor tool configuration has the following fields:

 - **mode**: can be "analyze", which will execute the model and output predictions and metrics, or "label", which will execute the model but write an updated labelled dataset with that output.
 - **threshold**: value between 0 and 1 that will convert a raw prediction into a classification.
 - **input**: section for input file and formats.
     - **dataset_class**: same as config field in Trainer tool.
     - **dataset**: relative path to a JSON file that will contain the dataset to predict on (which can be a full dataset or a reference dataset).
     - **base_dataset**: relative path to a JSON file for a dataset. If present, it means **dataset** is a reference dataset, and this is the base it is referencing. If absent, **dataset** is a full dataset.
     - **model**: relative path to the folder where the trained Tensorflow SaveModel to be used is stored.
     - **ts_model**: relative path to the folder where the trained time-series to be used is stored.
 - **output**: section for output files.
     - **predictions_output**: relative path to JSON file where the predictions will be stored. Only needed in "analyze" mode.
     - **metrics_output**: relative path to the JSON file where the metric information will be stored. Only needed in "analyze" mode.
     - **labelled_output**: relative path to JSON file where the labelled output will be stored. Only needed in "label" mode.
 - **time_interval**: time interval configuration.
   - **starting_interval**: time interval at which start aggregating the predictions for analyzing the results.
   - **interval_unit**: time interval unit. Possible values available here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
 - **metrics**: array containing objects describing the metrics to analyze. Only needed if "predict" mode is enabled. Each metric object contains:
   - **name**: a friendly name for the metric.
   - **type**: the metric type (can only be ErrorMetric or DistanceMetric).
   - **module**: name of the Python module inside the `harness/drifts` folder implementing this drift (see general README for more details).
   - **params**: dictionary containing drift module specific parameters.
        - For DistanceType metrics, it has to contain at least these parameters:
            - **distribution**: the distribution to use. Supported values are "normal" and "kernel_density". Another option is to use "custom" as a value, which means that the metric module will implement the actual density function (see general README for more details).
            - **range_start** and **range_end**: limits for the helper array of potential valid values for this distribution.
            - **range_step**: step for the helper array for the distribution.

The bash scripts for this tool include:
- **predictor.sh**: uses the `predictor_config.json` file, default for predicting and generating metrics.
- **labeller.sh**: uses the `labeller_config.json` file, uses the "label" mode to create an updated dataset with the predictions as its labels.
- **predictor_fullds.sh**: uses the `predictor_fullds_config.json` file, calculates only predictions for a full dataset.

## Merger Tool

This is a separate, simpler tool. Its configuration has the following fields:

 - **dataset1**: path to first JSON dataset to merge.
 - **dataset2**: path to second JSON dataset to merge.
 - **output**: path to JSON file with merged dataset.

The bash scripts for this tool include:
- **merger.sh**: uses the `merger_config.json` file, default for merging two datasets.
