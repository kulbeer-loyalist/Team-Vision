echo [$(date)]: "START"
echo [$(date)]: "Creating the environment with python version 3.10"
conda create --prefix ./venv python=3.10 -y
echo [$(date)]: "activating the environemnt"
conda  activate ./venv
echo [$(date)]: "installing the dev requiremets.txt"
pip install --no-cache-dir -r requirements.txt
echo [$(date)]: "END"