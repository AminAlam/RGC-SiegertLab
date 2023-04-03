# !/bin/bash

venv_path=venv

if [ -d "$DIR" ]; then
	echo "virtual env exists"
else
	python3 -m venv $venv_path
fi

source $venv_path/bin/activate
pip install -r requirements.txt
