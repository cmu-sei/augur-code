# augur-code
Code for Augur LENS

## Prerequisites
* Python 3.8 and pip3 are required.
  * Current version of Tensorflow (2.4) is not supported in Python 3.9.
  * If using MacOS, brew is recommended.
    * Install brew, if not installed: https://docs.brew.sh/Installation
    * Update brew, if needed: `brew update`
    * Install Python: `brew install python@3.8`
    * Add Python path to bash:
      * `echo "export PATH=\"/usr/local/opt/python@3.8/bin:$PATH\"" >> ~/.bash_profile`
* Pipenv is required.
  * `pip3 install --user pipenv`
  * Add pipenv path to bash:
    * `echo "export PATH=\"~/Library/Python/3.8/bin:$PATH\"" >> ~/.bash_profile`
    
## Setup
 * (Reference for Tensorflow: https://www.tensorflow.org/install/pip#virtual-environment-install)
 * From the project folder, run:
    * `pipenv install`

## Usage
  * From the project folder, run:
    * `pipenv shell`
  * Run the main scripts.
