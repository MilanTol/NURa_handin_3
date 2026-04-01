#!/bin/bash

# If you get a permission denied error for run.sh itself, run this line in the terminal:
# chmod +x ./run.sh

# If you get errors about weird (return) characters, and/or you edited run.sh on Windows, run this command:
# dos2unix ./run.sh &>/dev/null

# Make sure you do NOT run in a virtual environment (e.g. conda, uv), or your results may be different than when we run your code
# On the STRW computers, you may need to run "module purge" if you load any modules at startup
pythonversion="$( python3 --version | cut -d' ' -f2 )"
if [ "${pythonversion}" != "3.9.25" ] ; then
    echo "WARNING: Python version is different from the default vdesk one (${pythonversion} vs 3.9.25), this may or may not cause differences/errors."
fi
matplotlibversion="$( python3 -m pip list | grep "matplotlib " | tr -s ' ' | cut -d' ' -f2 )"
if [ "${matplotlibversion}" != "3.9.0" ] ; then
    echo "WARNING: Matplotlib version is different from the default vdesk one (${matplotlibversion} vs 3.9.0), this may or may not cause differences/errors."
fi
numpyversion="$( python3 -m pip list | grep "numpy " | tr -s ' ' | cut -d' ' -f2 )"
if [ "${numpyversion}" != "1.26.4" ] ; then
    echo "WARNING: Numpy version is different from the default vdesk one (${numpyversion} vs 1.26.4), this may or may not cause differences/errors."
fi

# Check if black formatter is installed
if ! python3 -m black --version &>/dev/null ; then
    echo "Black formatter not found. Installing..."
    python3 -m pip install black
fi

# Format all python files (note that this assumes your python files are all in the same directory as run.sh)
echo "Uniformly formatting Python code..."
python3 -m black .

echo "Clearing/creating the plotting directory..."
if [ ! -d "Plots" ]; then
  mkdir Plots
fi
rm -rf Plots/*

echo "Clearing/creating the code directory..."
if [ ! -d "Code" ]; then
  mkdir Code
fi
rm -rf Code/*

echo "Clearing/creating the calculations directory..."
if [ ! -d "Calculations" ]; then
  mkdir Calculations
fi
rm -rf Calculations/*

echo "Removing any PDFs..."
rm -f *.pdf

echo "Downloading data files..."
if [ ! -d "Data" ]; then
  mkdir Data
fi
cd Data
# Download the satellite galaxy data files if they don't exist yet. Make sure to exclude the .txt files from your handin, as they are large and we have them already
if [ ! -f "satgals_m11.txt" ]; then
    wget -q https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m11.txt
fi
if [ ! -f "satgals_m12.txt" ]; then
    wget -q https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m12.txt
fi
if [ ! -f "satgals_m13.txt" ]; then
    wget -q https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m13.txt
fi
if [ ! -f "satgals_m14.txt" ]; then
    wget -q https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m14.txt
fi
if [ ! -f "satgals_m15.txt" ]; then
    wget -q https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m15.txt
fi

# Move back to the main directory
cd ..

echo "Running Python script to fit the Number of satellite galaxies..."
python3 Q1_SatelliteGalaxies.py

# Copy the code to a text file which will be shown in the PDF
# ADAPT THIS, or in the tex load in only certain lines from these files relevant to the (sub)question!
cat Q1_SatelliteGalaxies.py > Code/satellites_maximize_code.txt
cat Q1_SatelliteGalaxies.py > Code/satellites_chi2_code.txt
cat Q1_SatelliteGalaxies.py > Code/satellites_poisson_code.txt
cat Q1_SatelliteGalaxies.py > Code/satellites_statistical_tests_code.txt
cat Q1_SatelliteGalaxies.py > Code/satellites_monte_carlo_code.txt

echo "Compiling LaTeX..."
pdflatex -interaction=batchmode NURA_handin_3.tex
pdflatex -interaction=batchmode NURA_handin_3.tex &>/dev/null # Run a second time to fix links/references

pdfs=($(ls -1 *.pdf 2> /dev/null))
if [[ "${#pdfs[@]}" -eq 0 ]] ; then
    echo "No PDF was produced; check the .log file for errors."
else
    echo "run.sh completed! Don't forget to hand in a *clean* version of this directory (remove the Data, Plots, Code and Calculations directories)."
fi
