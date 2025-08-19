source /Users/kgurashi/anaconda3/etc/profile.d/conda.sh
environment_name="mosaic_2"
elif [ -f "${environment_dir}/${environment_name}.yml" ]; then

# Define the directory where the environment file is located
environment_dir="/Users/kgurashi/GitHub/2024__EspressoPro_Manuscript/Scripts/Environments/${environment_name}"

# Remove the environment if it exists
if conda env list | awk '{print $1}' | grep -Fxq "$environment_name"; then
    echo "Removing existing $environment_name environment..."
    conda activate base
    conda env remove -n "$environment_name" -y
fi

# Create environment from file if YAML exists, else create from scratch
if [ -f "${environment_dir}/${environment_name}.yml" ]; then
    echo "Environment file found. Creating $environment_name from file..."
    conda env create -f "${environment_dir}/${environment_name}.yml"
else
    echo "Creating $environment_name environment from scratch..."
    conda create -n "$environment_name" -c missionbio -c conda-forge \
        python=3.10 \
        missionbio.mosaic-base=3.12.2 \
        python-kaleido -y
    conda activate "$environment_name"
    python -m pip install --upgrade pip
    conda install -n "$environment_name" ipykernel --update-deps --force-reinstall -y
    conda install -n "$environment_name" scanpy --update-deps --force-reinstall -y
    conda install -n "$environment_name" scikit-learn --update-deps --force-reinstall -y
    conda install -n "$environment_name" harmonypy --update-deps --force-reinstall -y
    conda install -n "$environment_name" cellhint --update-deps --force-reinstall -y
    python -m pip install notebook==6.5.4
    python -m pip install itables
    python -m pip install umap-learn==0.5.6
    python -m pip install chrov==0.0.3
    python -m pip install xgboost
    python -m pip install leidenalg
    python -m pip install venny4py
    python -m pip install netcal
fi
