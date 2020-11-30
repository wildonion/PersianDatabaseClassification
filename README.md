## Setup

* Create an environment: ```conda create -n uniXerr```
* Create the environment using the _scai.yml_ file: ```conda env create -f scai.yml```
* Activate _uniXerr_ environment: ```conda activate scai```
* Update the environment using _uniXerr.yml_ file: ```conda env update -f scai.yml --prune```
* Export your active environment to _uniXerr.yml_ file: ```conda env export | grep -v "^prefix: " > scai.yml```
