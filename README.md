# 🛰️ **reflspeckit**

> ⚙️ *A modern toolkit for working with any and all flavors of spectral data with a focus on applications for reflectance/emittance imaging spectroscopy*

---

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/reflspeckit)](https://pypi.org/project/reflspeckit/)
[![PyPI - License](https://img.shields.io/pypi/l/reflspeckit)](https://opensource.org/licenses/MIT)
[![PyPI - Version](https://img.shields.io/pypi/v/reflspeckit)](https://pypi.org/project/reflspeckit/)

---

## 🧠 What is `reflspeckit`?

`reflspeckit` is a lightweight, modular Python package designed to make analysis of **spectral data cubes** simple, flexible, and fun. Whether you're exploring planetary hyperspectral data, performing band analysis, or building your own spectral pipelines — this toolkit’s got you covered.

---

## ⚙️ Package Structure

`reflspeckit` provides two primary classes for the analysis of spectral data plus a third class specialized for large datasets:

>- 📄**Spec1D** - handles 1-dimensional, single spectrum data
>- 📒**Spec3D** - handles 3-dimensional spectral image cubes
>- 🗄️**StreamingSpec3D** - handles large image cubes using a streaming approach

Each class has equivalent methods, which are listed below:

---

## 🧰 Available Methods

|  Method | Description |
|-------------|-------------|
| 🚩`outlier_removal`| Removes **anomalous data** in the spectral domain |
| 🔊`noise_reduction` | Provides filtering methods to **smooth data** in the spectral domain |
| 📈`polyfit` | Performs a **least squares polynomial** fit over a specified spectral region |

## 💡 Spectral Utilities

Various spectral utilities are available through the `reflspeckit.utils` subpackge.

|  Module | Description |
|-------------|-------------|
| `get_nonzero`|     If you have an empty 3D image array with the first two dimensions being pixels and the third dimension of size N, and each pixel is filled in to a certain depth, M <= N, this function returns a 2D image array that picks out all the pixel values at position M. |

*More utilities coming soon! As a work through my Ph.D., I will add all the various utility functions I write for spectral data processing here!*

---

## 🚀 **Quick Start**

```bash
pip install reflspeckit
```

```python
import reflspeckit as rsk

# Loading in a single spectrum
my_spectrum = rsk.Spec1d(spectrum_array, wavelength_array)
my_spectrum.remove_outliers()
myspectrum.noise_reduction(method="box_filter", filter_width=5)
print(myspectrum.filtered)  # Contains filtered spectrum

# Loading in a spectral image cube
my_cube = rsk.Spec1d(cube_array, wavelength_array)
my_cube.remove_outliers()
my_cube.noise_reduction(method="box_filter", filter_width=5)
print(myspectrum.cube)  # Sequentially replaces myspectrum.cube to save memory.
```

## 🔗 Links

- **GitHub**: [https://github.com/z-vig/reflspeckit.git](https://github.com/z-vig/reflspeckit.git)
- **Docs**: (coming soon!)
