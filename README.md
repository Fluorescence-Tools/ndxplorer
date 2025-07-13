![conda build](https://github.com/fluorescence-tools/ndxplorer/actions/workflows/conda-build.yml/badge.svg)
[![Anaconda-Server Version](https://anaconda.org/tpeulen/ndxplorer/badges/version.svg)](https://anaconda.org/tpeulen/chisurf)

# ndxplorer

ndXplorer is an interactive tool for analyzing and visualizing multidimensional data sets. It provides a user-friendly graphical interface for data exploration, dimensionality reduction, clustering, and visualization.

## Overview

ndXplorer is designed to handle various types of multidimensional data, including:
* Multi-dimensional fluorescence (MFD) data
* Sampling of posterior distributions from [ChiSurf](https://github.com/Fluorescence-Tools/chisurf)
* Inter-label distance distributions from [Olga](https://github.com/Fluorescence-Tools/Olga)
* General multidimensional scientific data

![ndxplorer GUI][1]

ndxplorer can be used as a standalone application or integrated into other software as a tool.

## Features

* **Interactive Data Visualization**: Create and customize various plot types for multidimensional data
* **Dimensionality Reduction**: Apply UMAP (Uniform Manifold Approximation and Projection) for visualizing high-dimensional data
* **Clustering Analysis**: Perform clustering using algorithms like HDBSCAN and K-means
* **Data Import/Export**: Read and write data in various formats
* **Curve Overlay**: Overlay curves on plots for comparison
* **Parameter Editing**: Interactive editing of visualization parameters
* **Code Editing**: Built-in code editor for customization
* **Customizable UI**: Flexible user interface with theme support

## Dependencies

ndXplorer requires the following dependencies:
* Python
* PyQt and QtPy (for GUI)
* NumPy, pandas, scipy (for data handling and computation)
* matplotlib and PyQtGraph (for plotting)
* UMAP-learn (for dimensionality reduction)
* HDBSCAN and scikit-learn (for clustering)
* guidata and guiqwt (for GUI components)

## Installation

### Using conda (recommended)

The easiest way to install ndXplorer is via conda:

```bash
conda install -c tpeulen ndxplorer
```

### From source

To install from source:

```bash
git clone https://gitlab.peulen.xyz/tpeulen/ndxplorer.git
cd ndxplorer
pip install -e .
```

## Usage

### Starting ndXplorer

To start ndXplorer, simply run:

```bash
ndxplorer
```

### Basic Workflow

1. **Load Data**: Import your multidimensional data
2. **Explore**: Use the interactive plots to explore your data
3. **Analyze**: Apply dimensionality reduction and clustering techniques
4. **Visualize**: Create custom visualizations of your data
5. **Export**: Save your results and visualizations

## Contributing

Contributions to ndXplorer are welcome! Please feel free to submit a Pull Request.

## License

ndXplorer is licensed under the GPL 2.1 License. See the LICENSE file for details.

[1]: doc/gui.png "ndxplorer GUI"
