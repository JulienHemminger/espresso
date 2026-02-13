# Build
https://espressomd.github.io/doc/installation.html#quick-installation

cd build
rm CMakeCache.txt

cmake .. -DEXTERNAL_CUDA_COMPUTE_CAPABILITY=6.1 \
         -DCMAKE_BUILD_TYPE=Release \
         -DPYTHON_EXECUTABLE=$(which python)

make -j$(nproc)


# Run Python Scripts (Espresso already built)
* open terminal at /home/main/Documents/Career/1_Studium/espresso
* source espresso_env/bin/activate
* cd build
* ./pypresso -m jupyter lab


