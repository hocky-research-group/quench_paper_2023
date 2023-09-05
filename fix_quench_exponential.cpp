/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "stdio.h"
#include "string.h"
#include "fix_quench_exponential.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "mpi.h"
#include "math.h"
#include "comm.h"
#include "input.h"
#include "variable.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "group.h"
#include "math_extra.h"
#include "compute.h"
#include "domain.h"
#include "group.h"
#include "universe.h"
#include <time.h>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixQuenchExponential::FixQuenchExponential(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{

// fix 1 all baoab  gamma  temp  seed 

    
  if (narg < 4) error->all(FLERR,"Illegal fix quench exponential command :: too few args");
  
   
  gamma = force->numeric(FLERR,arg[3]);
  if(gamma<0.0){
      if (screen) fprintf(screen, ">Friction coeff is negative, running backwards in time with gamma %e", gamma);
      if (logfile) fprintf(logfile, ">Friction coeff is negative, running backwards in time with gamma %e", gamma);
      gamma = fabs(gamma);
      my_dt = -update->dt;
  }
  else {
      my_dt = update->dt;
  }
   
}

FixQuenchExponential::~FixQuenchExponential()
{
}

/* ---------------------------------------------------------------------- */

int FixQuenchExponential::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixQuenchExponential::init()
{

  double boltz = force->boltz;
  double mvv2e = force->mvv2e;
  double ftm2v = force->ftm2v;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  
  dtv = 0.5 * my_dt;
  dtf = 0.5 * my_dt * force->ftm2v;
  
  c1 = exp(-my_dt * gamma);
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixQuenchExponential::initial_integrate(int vflag)
{
  double dtfm;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double sqrm;
  int tdim;
  tagint *tag = atom->tag;
  double R[ universe->nprocs ] ;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  double ftm2v = force->ftm2v;
  double tv, stats[5];
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int ii,jj;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
 
  
  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
	sqrm = 1.0 / sqrt( rmass[i] ); 
        dtfm = dtf / rmass[i];
	// B
	  v[i][0] += dtfm * f[i][0];
	  v[i][1] += dtfm * f[i][1];
	  v[i][2] += dtfm * f[i][2];
	// A
	  x[i][0] += dtv * v[i][0];
	  x[i][1] += dtv * v[i][1];
	  x[i][2] += dtv * v[i][2];
	// O
	  v[i][0] = c1 * v[i][0];
	  v[i][1] = c1 * v[i][1];
	  v[i][2] = c1 * v[i][2];
	// A
	  x[i][0] += dtv * v[i][0];
	  x[i][1] += dtv * v[i][1];
	  x[i][2] += dtv * v[i][2];
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
	sqrm = 1.0 / sqrt( mass[type[i]] ); 
        dtfm = dtf / mass[type[i]];
	// B
	  v[i][0] += dtfm * f[i][0];
	  v[i][1] += dtfm * f[i][1];
	  v[i][2] += dtfm * f[i][2];
	// A
	  x[i][0] += dtv * v[i][0];
	  x[i][1] += dtv * v[i][1];
	  x[i][2] += dtv * v[i][2];
	// O
	  v[i][0] = c1 * v[i][0];
	  v[i][1] = c1 * v[i][1];
	  v[i][2] = c1 * v[i][2];
	// A
	  x[i][0] += dtv * v[i][0];
	  x[i][1] += dtv * v[i][1];
	  x[i][2] += dtv * v[i][2];
      }
  }
  
  
   
  
}

/* ---------------------------------------------------------------------- */

void FixQuenchExponential::final_integrate()
{
  double dtfm;

  // update v of atoms in group

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
  }
}

void FixQuenchExponential::reset_dt()
{
  dtv = 0.5 * my_dt;
  dtf = 0.5 * my_dt * force->ftm2v;
}
