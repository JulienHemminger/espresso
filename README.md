# Planning

* remove unnessecary files, merge duplicate files, rename them, etc





* repair broken/false plots (where custom != legacy)


* cleanup source code (remove unnessecary comments, etc)

  * make scripts re-use "save_figure_w_timestamp" or "run_lerp\_plot"


* cleanup outputs (plot labels, colors, print statements)

  * maybe: make most plots share same color sceme

# Readme

### Build

<https://espressomd.github.io/doc/installation.html#quick-installation>

cd build
rm CMakeCache.txt

cmake .. -DEXTERNAL\_CUDA\_COMPUTE\_CAPABILITY=6.1 -DCMAKE\_BUILD\_TYPE=Release -DPYTHON\_EXECUTABLE=\$(which python)

make -j\$(nproc)

### Run Python Scripts (Espresso already built)

./pypresso /home/main/Documents/Career/1\_Studium/Semester/9\_WS\_25/BArbeit/bsc\_hemminger/elcic/energy/dual\_plates/neutral/dipole/dual\_plate\_zshift.py
