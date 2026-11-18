# PathSnap

GPS track exploration and preprocessing for the ML hackathon.

- `notebooks/gps_preprocessing.ipynb`: archive checks, road graph validation, GPS cleanup and map views.
- `data/Student folder.zip`: local copy of the supplied development dataset (not committed).

Install NumPy, pandas, SciPy, Matplotlib, Folium and Jupyter in your Python environment. Start Jupyter from the project root or `notebooks/`, open the notebook and run the cells in order.

For a dataset stored elsewhere, set `PATHSNAP_DATA_ZIP` to its path before starting Jupyter. The notebook still accepts `Student folder.zip` in the working directory for existing setups. Map tiles require internet access.

This repository currently covers preprocessing; model training and route reconstruction are still to be implemented.
