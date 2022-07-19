#drivers-of-peak-warming

conda create --name drivers-of-peak-warming

conda activate drivers-of-peak-warming

conda install pip

pip install -r requirements.txt

## To install FaIRv2.0.0-alpha
- Download the FaIRv2.0.0-alpha.zip file from Zenodo (https://doi.org/10.5281/zenodo.4683173) and save it
- Use the command "unzip FaIRv2.0.0-alpha.zip" to unzip the downloaded model
- Navigate to inside the extracted folder ("cd FAIRv2.0.0-alpha")
- Use the command "pip install -e ./" to install fair as a local python module to your machine (on the fair_env python environment)
