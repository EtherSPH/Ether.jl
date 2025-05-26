# Ether.jl

A particle-based simulation framework running on both cpu and gpu.

# Dependencies

This project is built on top of the following packages:

- [Atomix.jl](https://github.com/JuliaConcurrent/Atomix.jl): atomic operations
- [CodecZstd.jl](https://github.com/JuliaIO/CodecZstd.jl): compress the `.jld` files in a balanced mode between speed and compression-ratio
- [Dates.jl](https://docs.julialang.org/en/v1/stdlib/Dates/#:~:text=The%20Dates%20module%20provides%20two%20types%20for%20working,respectively%3B%20both%20are%20subtypes%20of%20the%20abstract%20TimeType.): get time-stamp
- [FIGlet.jl](https://github.com/kdheepak/FIGlet.jl): customize the project's logo
- [JLD2.jl](https://github.com/JuliaIO/JLD2.jl): store and load raw data in `h5` format
- [JSON.jl](https://github.com/JuliaIO/JSON.jl): configure the simulation in `json` format
- [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl): format the code
- [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl): an abstraction layer for parallel kernels, which is the core dependency of this package
- [OrderedCollections.jl](https://github.com/JuliaCollections/OrderedCollections.jl): don't break the configuration order
- [ProgressMeter.jl](https://github.com/timholy/ProgressMeter.jl): show the progress of tasks
- [WriteVTK.jl](https://github.com/JuliaVTK/WriteVTK.jl): export calculation result as `.vtp` format
- [YAML.jl](https://github.com/JuliaData/YAML.jl): configure the simulation in `yaml` format

To run the code on gpu, one of the following backends is required:

- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl): NVIDIA discrete GPUs
- [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl): AMD GPUs, both discrete and integrated
- [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl): Intel GPUs, both discrete and integrated
- [Metal.jl](https://github.com/JuliaGPU/Metal.jl): Apple M-series GPUs

Scripts to build these backends are included in `script/build` folder. Supposing you are a macOS user, you can run the following command to build the `Metal.jl` backend:

```bash
julia script/build/metal.jl
```

Some memos:

- 2025.05.19: currently fix [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) to version "2.0.2"
- ...