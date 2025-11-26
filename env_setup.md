### Setup autodocking vida conda environment

```
$ conda create -n vina python=3.11
$ conda activate vina
$ conda config --env --add channels conda-forge
```

```
$ conda install -c conda-forge numpy swig boost-cpp libboost sphinx sphinx_rtd_theme
$ pip install vina
```

### Download the vina executable from github
Github link [https://github.com/ccsb-scripps/AutoDock-Vina/releases]

```
$ wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64
$ chmod +x vina_1.2.7_linux_x86_64
$ mv vina_1.2.7_linux_x86_64 $CONDA_PREFIX/bin/vina
$ which vina
$ vina --version
```

### Download meeko
```
pip install -U scipy rdkit meeko gemmi prody
```