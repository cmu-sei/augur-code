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
 * From the project folder, run:
    * `pipenv install`
    * (For a manual Tensorflow setup, see: https://www.tensorflow.org/install/pip#virtual-environment-install)

### Usage
  * From the tools folder, run the tool that you want to run:
    * `bash <tool>.sh`
    * Where <toool> is the name of one of the tools in that folder (i.e., "trainer", "drifter", etc.).

## Containerized Version
### Prerequisites
* Docker and Docker-Compose are required.
* If working behind a proxy, Docker needs to be configured to 1) download images and 2) so that images that are being built can download components from the Internet.

### Setup
* From the project folder, run:
  * `bash build_container.sh`

### Usage
* From the project folder, run:
  * `bash run_compose.sh <tool_name>`
  * Where <tool> is the name of one of the tools in the tools folder; i.e., "trainer", "labeller", etc.
  * Note that if you stop this with Ctrl+C, the process doesn't stop, it just runs in the background, and you can get back to viewing its output with `bash logs_compose.sh` 
* To stop, run:
  * `bash stop_compose.sh`
