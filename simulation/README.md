# Simulator
Simulates and saves raw point cloud data for nuclear pore complexes (NPCs), vesicles, microtubules, actins, and mitochondria. Edit the configuration for each of the structures in `config.py`. 

## ThunderSTORM Simulation
The point cloud generated with the simulator can be used with a PSF simulator such as testSTORM to get an SMLM stack which you can then process using ThunderSTORM. Alternatively, this simulator also provides the ability to simulate the effect of ThunderSTORM directly. This option is enabled by default, and you will find the ThunderSTORM-simulated point clouds in the columns 'x', 'y', and 'z'. ThunderSTORM simulation is done by adding a random deviation to all the points, then adding a foreground noise (closer to the structures) and finally a background noise (all over the available space). Edit the noise level and other parameters for ThunderSTORM simulation in `storm/config.py`.


## Usage
Here is the complete help documentation for the simulator:

```bash
usage: datagen.py [-h] [-n N] [--with-normals] [--no-storm] [-o O] [-f FTYPE] [--workers WORKERS] [--overwrite] [--header] [--seed SEED] structure

Simulate a structure and save the points as a csv/parquet file.

positional arguments:
  structure             The structure to simulate. Available options: vesicle, npc, mito, actin, microtubules, all

optional arguments:
  -h, --help            show this help message and exit
  -n N                  Number of structures to simulate.
  --with-normals        Whether to include the normals in the output file.
  --no-storm            Specify to NOT add thunderstorm noise to the points.
  -o O                  The output directory prefix for the files. The files will be saved as {prefix}/{structure_name}_{i}.[csv/parquet].Defaults to 'data/{ddmmyy-hhmms}'.
  -f FTYPE, --ftype FTYPE
                        The file type to save the points as. Available file types: csv, parquet. Defaults to parquet.
  --workers WORKERS     Number of workers to use for multiprocessing.
  --overwrite           Whether to overwrite the output file if it already exists.
  --header              Whether to include the header row in the file.
  --seed SEED           The random seed to use. Defaults to None, which does not set the seed.

```

## Output Format

The data is saved as either a csv file or a parquet file (default, more space-efficient than csv). The files can be loaded using the `pandas` library (`read_csv` for csv files and `read_parquet` for parquet files). Each of the saved files have one or more of the following sets of columns (depending on the options provided):

1. `x_gt`, `y_gt`, `z_gt`: The 3D coordinates of the simulated points on the surface of the structure(s) (Ideal points). You can use these with other PSF simulators to get an SMLM stack, and perform SMLM reconstruction by yourself.
2. `label`: The label of the structure that the point belongs to. If `--no-storm` is *not* specified, also includes the label for whether the point belongs to foreground (`noise_fg`) or background noise (`noise_bg`).
3. `instance_id`: The instance ID of the structure that the point belongs to. Multiple instances of the same structure are simulated, and each instance is assigned a unique ID.
4. `x`, `y`, `z`: The 3D coordinates of the ThunderSTORM-simulated points (if `--no-storm` is *not* specified). These are the points with added noise to simulate the effect of ThunderSTORM.
5. `nx`, `ny`, `nz`: The 3D coordinates of the normals of the points (if `--with-normals` is specified).
6. `theta`, `phi`: The spherical coordinates of the surface normals (if `--with-normals` is specified), in radians. These two columns can be mathematically computed from `nx`, `ny`, `nz` and are provided just for convenience. The interconversion formula used is: 
```python
    nx = sin(theta) * cos(phi)
    ny = sin(theta) * sin(phi)
    nz = cos(theta)
```


# Examples

```bash
python datagen.py npc -n 1000 --with-normals -o data/v2 --workers 8 --overwrite
```
Generates 1000 nuclear pore complex (NPC) structures, with 8 parallel workers and saves them to `data/v2/`.

---

```bash
python datagen.py vesicle -n 1000 --with-normals -o data/v2 --workers 8 --overwrite --no-storm
```
Generates 1000 vesicle structures, with 8 parallel workers and saves them to `data/v2/`. Does not add ThunderSTORM noise to the points.

---

```bash
cd example_scripts
python plot_normals_orthonormals_noodles.py
```
Plots an example mitochondrion simulation. The first plot shows the orthonormal vectors and cross section normals. Close the first plot to reveal the second plot, which shows surface normals. Helpful to visualize working mechanisms behind the simulator. More information in the thesis Appendix.