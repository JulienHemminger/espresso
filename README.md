# Planning

* remove unnessecary files
  * can i remove any plots/tests?

* get all plots to work

* merge duplicate files, rename them, etc

# Readme

### Build

<https://espressomd.github.io/doc/installation.html#quick-installation>

cd build
rm CMakeCache.txt

cmake .. -DEXTERNAL\_CUDA\_COMPUTE\_CAPABILITY=6.1 -DCMAKE\_BUILD\_TYPE=Release -DPYTHON\_EXECUTABLE=\$(which python)

make -j\$(nproc)

### Run Python Scripts (Espresso already built)

./pypresso /home/main/Documents/Career/1\_Studium/Semester/9\_WS\_25/BArbeit/bsc\_hemminger/elcic/energy/dual\_plates/neutral/dipole/dual\_plate\_zshift.py
