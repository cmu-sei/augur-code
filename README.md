# augur-code
Code for Augur LENS

## Non-Containerized Version
### Prerequisites
* Python 3.8 and pip3 are required (Tensorflow 2.4 is not supported in Python 3.9).
  * If using MacOS, brew is recommended.
    * Install brew, if not installed: https://docs.brew.sh/Installation
    * Update brew, if needed: `brew update`
    * Install Python: `brew install python@3.8`
    * Add Python path to bash:
      * `echo "export PATH=\"/usr/local/opt/python@3.8/bin:$PATH\"" >> ~/.bash_profile`
* Pipenv is required.
  * `pip3 install --user pipenv`
  * In MacOS, add pipenv path to bash:
    * `echo "export PATH=\"~/Library/Python/3.8/bin:$PATH\"" >> ~/.bash_profile`
    
### Setup
 * From the `tools/` folder, run:
    * `pipenv install`
    * (For a manual Tensorflow setup, see: https://www.tensorflow.org/install/pip#virtual-environment-install)

### Usage
  * From the `tools/` folder, run the tool that you want to run:
    * `bash <tool>.sh [params]`
    * Where 
      * "tool" is the name of one of the tools in that folder (i.e., "trainer", "drifter", etc.).
     * [params]: optional, it can be either:
        * "--config", followed by a config file name (path relative to current folder)
        * "--exp_config" to use experiment configurations, followed by a config file name and path relative to the experiments folder.
        * "--exp_user": a menu is presented to select from config files in the experiments folder.
        * "--test": runs the tool in test mode (if available).
        * "--store": for the Predictor, generates a folder and a zip file in the output/packaged subfolder contained information about the run. More specifically, it contains 1) predictions JSON output, 2) metrics JSON output, 3) drifted ids JSON input, 4) trained model input, 5) log, 6) config file used.

## Containerized Version
### Prerequisites
* Docker and Docker-Compose are required.
* If working behind a proxy, Docker needs to be configured to 1) download images and 2) so that images that are being built can download components from the Internet.

### Setup
* From the project folder, run:
  * `bash build_container.sh`

### Usage
* From the project folder, run:
  * `bash run_compose.sh <tool_name> [params]`
  * Where 
    * <tool_name> is the name of one of the tools in the tools folder; i.e., "trainer", "labeller", etc.
    * [params]: see explanation in "Non-Containerized Version" above.  
  * Note that if you stop this with Ctrl+C, the process doesn't stop, it just runs in the background, and you can get back to viewing its output with `bash logs_compose.sh` 
* To stop, run:
  * `bash stop_compose.sh`

### Extending
#### Drift Modules
New drift modules can be added to the harness/drifts folder. Each module only needs to have one function, with the following signature:

`def get_bin_index(sample_index, timebox_id, curr_bin_idx, num_total_bins, params):`

This function should return the index of the bin to select the next sample from, given the current sample index, the current timebox id, the current bin being used, the total number of bins, and whatever additional params are needed.

#### Metric Modules
New metric modules can be added to the harness/metrics folder. Each module needs to implement multiple functions, depending on the metric type.

For Distance Metrics, two functions are needed (one is optional):

 - `def metric_distance(p, q)`: Calculates a distance value between the two given probability distributions (numpy arrays).
 - `def metric_density(data, valid_range, density_params, config_params)` - OPTIONAL: Calculates the probability density function for the given data (a numpy array), with the given valid_range of potential valid values, with the available density params (a dictionary), and the optional configuration params (a dictionary). The "distribution" config value has to be set to "custom" for this to be used (see README in tools\).

 For Error Metrics, at least one function is needed:

  - `def metric_error(time_interval_id, time_series, ts_predictions)`: Calculates the error for the given time interval, knowing the aggregated data from the time_series, and the time series predictions as a parameter.

#### Dataset Structures
New dataset structures should be added to the harness/datasets folder, to a subfolder with the name of the new structure. Two modules should be added in there: a dataset module and a model module. The recommended naming structure is:

 - Folder: harness/datasets/<datasetname>
 - Dataset module: harness/datasets/<datasetname>/<datasetname>_dataset.py
 - Model module: harness/datasets/<datasetname>/<datasetname>_model.py

The dataset module needs to implement a class that derives from `dataset.DataSet`, and that overrides or implements the following methods, as needed. Overrides should call the super method first.
 - `def load_from_file(self, dataset_filename)`: (override) loads the dataset from the given JSON file.
 - `def save_to_file(self, output_filename)`: (override) stores the dataset to the given JSON file.
 - `def get_sample(self, position)`: (override) returns a sample as a dictionary.
 - `def allocate_space(self, size)`: (override) pre-sets enough space for a dataset, useful when creating a full one from a reference, to make it work fast.  
 - `def add_sample(self, position, sample)`: (override) adds a sample received as a dictionary to the given position.
 - `def get_model_input(self)`: returns the input needed for the corresponding model.
 - `def get_single_input(self)`: if model has multiple inputs, returns only one of them (any). Otherwise, returns the same as get_model_input.
 - `def get_output(self)`: returns the labelled output of the model.
 - `def set_output(self, new_output)`: sets the labelled output of the model.

The model module needs to implement the following three functions:
 - `def create_model()`: returns a Tensorflow model for this dataset.
 - `def split_data(dataset, validation_percentage)`: given a dataset and the % to use for validation, returns a TrainingSet object with the separated data.
 - `def get_fold_data(dataset, train_index, test_index)`: given a dataset, and a train and test index, returns a TrainingSet object with the given indexes as a subset of the dataset.
