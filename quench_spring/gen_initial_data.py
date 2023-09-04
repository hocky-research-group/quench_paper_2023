import numpy as np
import os

import yaml,sys
if len(sys.argv) != 2 or sys.argv[1][-5:] != ".yaml":
    print("Usage: %s yaml_file"%(sys.argv[0]))
    sys.exit()
with open(sys.argv[1],'r') as f:
    parameters = yaml.full_load(f)
    high_T_params = parameters["high_T"]
    globals().update(high_T_params)

import lammps
for N in N_list:
    lmp = lammps.lammps()
    lmp.command("units           lj")
    lmp.command("dimension       3")
    lmp.command("atom_style      atomic")
    lmp.command("boundary        p p p")
    lmp.command("region          box block 0 10 0 10 0 10")
    lmp.command("create_box      1 box")
    lmp.command("mass            1 1.0")
    lmp.command("timestep        0.001")
    lmp.command("create_atoms    1 random %d 11591538 box"%(N))
    lmp.command("set type 1 vx 1.0 vy 1.0 vz 1.0")
    lmp.command("velocity        all create 1.0 4928459 dist gaussian")
    lmp.command("write_data spring_initial_%d.data"%(N))

