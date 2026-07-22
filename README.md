# xFormers-for-SAIL

> **Important Notice**: The operators and features on the current branch only support **Zhenwu 810 and Zhenwu 810E**. Calling the relevant operators (cutlassF and cutlassB) on other series will throw an error. When encountering this error, please confirm that you are running in the correct environment.
>
> **Note**: This xFormers-for-SAIL repository exists as a **submodule of PyTorch-for-SAIL** and only provides **a subset of features** compared to the main [xFormers repository](https://github.com/facebookresearch/xformers). It is designed specifically for use with **PyTorch-for-SAIL** backend and may not include all features available in the upstream xFormers project.

## Building and Installing

### Building with PyTorch-for-SAIL

When building PyTorch-for-SAIL with PPU support, xFormers-for-SAIL will be built automatically as a static library. Follow the standard PyTorch-for-SAIL build process.

### Building Standalone

To build xFormers-for-SAIL as a standalone package:

```bash
cd /path/to/xformers
python setup.py bdist_wheel
```

This will create a wheel package that can be installed with `pip install dist/*.whl`.

### Supported Features

This version of xFormers-for-SAIL supports following features for memory efficient attention:

- **cutlassF** (forward kernel)
- **cutlassB** (backward kernel)

### Running Tests

Copy the test files to a separate directory to avoid reference directory issues:

```bash
cp -r tests /tmp/xformers_test
cd /tmp/xformers_test
python -m pytest test_mem_eff_attention.py::test_forward -v
python -m pytest test_mem_eff_attention.py::test_backward -v
```
