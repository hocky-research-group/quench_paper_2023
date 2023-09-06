def run_baoab_spring(command_file,input_file,N,K,dt,thermo_freq,run_temp,run_gamma,eq_steps,log_prefix,nrestarts=1):
    """
        run NVT and then quench for N harmonic springs system
    """
    import lammps
    lmp = lammps.lammps()
    log_file = log_prefix + "N%d_K%.1f_rt%.1f_rg%.2e_eqsteps%d.log"%(N,K,run_temp,run_gamma,eq_steps)
    lmp.command("log %s"%(log_file))
    for command in open(command_file,"r").readlines():
        lmp.command(command.strip())
    lmp.command("read_data %s"%(input_file))
    lmp.command("timestep %f"%(dt))
    lmp.command("fix 1 all spring/self %.1f xyz"%(K))
    lmp.command("fix 2 all langevin %f %f %f 48279"%(run_temp,run_temp,1.0/run_gamma))
    lmp.command("fix 3 all nve")
    lmp.command("fix_modify      1 energy yes")
    lmp.command("thermo_style    custom step temp ke pe etotal")
    lmp.command("thermo_modify norm no")
    lmp.command("thermo %d"%(eq_steps//10))
    lmp.command("run %d"%(eq_steps))
    lmp.command("reset_timestep 0")
    restart_file = log_file.replace(".log","_step*.restart")
    lmp.command("restart %d %s"%(thermo_freq,restart_file))
    lmp.command("thermo %d"%(thermo_freq))
    lmp.command("run %d"%(nrestarts*thermo_freq))
    return log_file

def run_quench_spring(command_file,restart_file,log_prefix,dt,K,quench_gamma,quench_thermo_freq,quench_steps):
    """
        run quench
    """
    import lammps
    lmp = lammps.lammps()
    log_file = log_prefix + "qg%.2e.log"%(quench_gamma)
    lmp.command("log %s"%(log_file))
    for command in open(command_file,'r').readlines():
        lmp.command(command.strip())
    lmp.command("read_restart %s"%(restart_file))
    if quench_gamma < 0:
        lmp.command("variable vx atom -vx")
        lmp.command("variable vy atom -vy")
        lmp.command("variable vz atom -vz")
        lmp.command("velocity all set v_vx v_vy v_vz")
    lmp.command("reset_timestep 0")
    lmp.command("timestep %f"%(dt))
    lmp.command("fix 1 all spring/self %.1f xyz"%(K))
    lmp.command("fix_modify      1 energy yes")
    lmp.command("thermo_style    custom step temp ke pe etotal")
    lmp.command("thermo_modify norm no")
    lmp.command("thermo %d"%(quench_thermo_freq))
    # quench
    lmp.command("fix 2 all quench_exponential %f"%(quench_gamma))
    # check completion
    restart_file = log_file.replace(".log","_step*.restart")
    lmp.command("restart %d %s"%(quench_steps,restart_file))
    lmp.command("run %d"%(quench_steps))
    return log_file

def get_thermo_data(log_file):
    import lammps
    return lammps.get_thermo_data(open(log_file,'r').read())[-1].thermo

def log_sub_binary(lnA,lnB):
    import numpy as np
    return lnA + np.log(1.0 - np.exp(lnB - lnA))

def log_sum_binary(lnA,lnB):
    import numpy as np
    if lnA == -np.inf:
        return lnB 
    if lnB == -np.inf:
        return lnA 
    if lnA >= lnB:
        return lnA + np.log(1.0+np.exp(lnB-lnA))
    else:
        return lnB + np.log(1.0+np.exp(lnA-lnB))

def log_sum(exp_part_list):
    import numpy as np
    if len(exp_part_list) == 0:
        return -np.inf
    exp_part_list = np.sort(exp_part_list)[-1::-1]
    result = exp_part_list[0].copy()
    for exp_part in exp_part_list[1:]:
        if exp_part == -np.inf:
            continue
        elif result == -np.inf:
            result = exp_part.copy()
        else:
            result = result + np.log(1.0+np.exp(exp_part-result))
    return result

def find_Ebound(log_file,index,heat=True):
    Emin = get_thermo_data(log_file).TotEng[-1]
    if heat:
        Emax = get_thermo_data(log_file.replace("qg","qg-")).TotEng[-1]
    else:
        Emax = get_thermo_data(log_file).TotEng[0]
    return index,Emin,Emax

def infinite_stopping_compute_lnQ(log_file,target_kbt,run_kbt,dof,dt,quench_gamma,Emin,Emax,index,heat=True):
    import lammps
    import numpy as np
    # merge
    thermo_data = get_thermo_data(log_file)
    etot_trj = np.array(thermo_data.TotEng)
    t_trj = np.array(thermo_data.Step)
    if heat:
        log_b_file = log_file.replace("qg","qg-")
        thermo_data_b = get_thermo_data(log_b_file)
        etot_trj_b = np.array(thermo_data_b.TotEng)[1:]
        origin = len(etot_trj_b)
        etot_trj = np.append(np.flip(etot_trj_b),etot_trj)
        t_trj_b = np.array(thermo_data_b.Step)[1:] * -1.0 # change this if dt_b changes
        t_trj = np.append(np.flip(t_trj_b),t_trj) * dt
    dgt_trj = t_trj * dof * quench_gamma
    ln_numerator_trj = -etot_trj / target_kbt - dgt_trj
    ln_denominator_trj = -etot_trj / run_kbt - dgt_trj
    # find tau^- and tau^+
    Emin_ind = np.argwhere(etot_trj < Emin)
    Emax_ind = np.argwhere(etot_trj > Emax)
    if len(Emax_ind) == 0:
        start = 0
    else:
        start = np.max(Emax_ind) + 1
    if len(Emin_ind) == 0:
        end = len(etot_trj) - 1
    else:
        end = np.min(Emin_ind) - 1
    ln_denominator = log_sum(ln_denominator_trj[start:end+1])
    lnQ = log_sum(ln_numerator_trj[start:end+1]) - ln_denominator
    time = t_trj[end] - t_trj[start]
    return index,lnQ,time



