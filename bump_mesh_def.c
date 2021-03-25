/*   Program bump_mesh_def.c    */
/*
Creates simple 3-D structured grid and writes it to a
CGNS file.

Example compilation for this program is (change paths!):

gcc bump_mesh_def.c -L $CGNS_HOME/lib -I $CGNS_HOME/include -o bump_mesh_def

(../CGNS_CVS/cgnslib/LINUX/ is the location where the compiled
library libcgns.a is located)
*/

#include <stdio.h>
#include <string.h>
#include <math.h>
/* cgnslib.h file must be located in directory specified by -I during compile: */
#include "cgnslib.h"

#if CGNS_VERSION < 3100
# define cgsize_t int
#else
# if CG_BUILD_SCOPE
#  error enumeration scoping needs to be off
# endif
#endif

double xDistFunc(int ind, int imax, double lX);

double zDistFunc(int ind, int imax, double lX);


int main()
{
/*
   dimension statements (note that tri-dimensional arrays
   x,y,z must be dimensioned exactly as [N][17][21] (N>=9)
   for this particular case or else they will be written to
   the CGNS file incorrectly!  Other options are to use 1-D
   arrays, use dynamic memory, or pass index values to a
   subroutine and dimension exactly there):
*/

   
   cgsize_t isize[3][3];
   cgsize_t ipnts[3][3];
   int ni,nj,nk,i,j,k;
   int index_file,icelldim,iphysdim,index_base;
   int index_zone,index_coord;
   char basename[33],zonename[33];

/* create gridpoints for simple example: */
   double lX = 3.0;
   double lY = 1.0;
   double lZ = 1.0;
   ni=97;
   nj=2;
   nk=49;
   double x[nk*nj*ni],y[nk*nj*ni],z[nk*nj*ni];
   for (k=0; k < nk; ++k)
   {
     for (j=0; j < nj; ++j)
     {
       for (i=0; i < ni; ++i)
       {
         x[k*nj*ni + j*ni + i]=xDistFunc(i, ni, lX);//(i/((double)ni-1.))*lX;//
         y[k*nj*ni + j*ni + i]=(j/((double)nj-1.))*lY;
         z[k*nj*ni + j*ni + i]=zDistFunc(k, nk, lZ);//(k/((double)nk-1.))*lZ;
       }
     }
   }
   printf("\ncreated simple 3-D grid points");

/* WRITE X, Y, Z GRID POINTS TO CGNS FILE */
/* open CGNS file for write */
   if (cg_open("grid_c.cgns",CG_MODE_WRITE,&index_file)) cg_error_exit();
/* create base (user can give any name) */
   strcpy(basename,"Base");
   icelldim=3;
   iphysdim=3;
   cg_base_write(index_file,basename,icelldim,iphysdim,&index_base);
/* define zone name (user can give any name) */
   strcpy(zonename,"Zone  1");
/* vertex size */
   isize[0][0]=ni;
   isize[0][1]=nj;
   isize[0][2]=nk;
/* cell size */
   isize[1][0]=isize[0][0]-1;
   isize[1][1]=isize[0][1]-1;
   isize[1][2]=isize[0][2]-1;
/* boundary vertex size (always zero for structured grids) */
   isize[2][0]=0;
   isize[2][1]=0;
   isize[2][2]=0;
/* create zone */
   cg_zone_write(index_file,index_base,zonename,*isize,Structured,&index_zone);
/* write grid coordinates (user must use SIDS-standard names here) */
   cg_coord_write(index_file,index_base,index_zone,RealDouble,"CoordinateX",
       x,&index_coord);
   cg_coord_write(index_file,index_base,index_zone,RealDouble,"CoordinateY",
       y,&index_coord);
   cg_coord_write(index_file,index_base,index_zone,RealDouble,"CoordinateZ",
       z,&index_coord);
/* close CGNS file */
   cg_close(index_file);
   printf("\nSuccessfully wrote grid to file grid_c.cgns\n");
   
   //boundary conditions
   if (cg_open("grid_c.cgns",CG_MODE_MODIFY,&index_file)) cg_error_exit();
   index_base=1;
   index_zone=1;
   cg_zone_read(index_file,index_base,index_zone,zonename,*isize);
   int ilo, ihi, jlo, jhi, ihi1, ihi2, klo, khi;
   int index_bc;
   ilo=1;
   ihi=isize[0][0];
   jlo=1;
   jhi=isize[0][1];
   klo=1;
   khi=isize[0][2];
   double fr1 = 1./8.;
   double fr2 = 1./6.;
   double fr3 = 3./6.;
   double fr4 = 5./6.;
   double fr5 = 7./8.;
   ihi1 = (int)(fr1*ihi);
   ihi2 = (int)(fr5*ihi);
   //inlet face
   ipnts[0][0]=ilo;
   ipnts[0][1]=jlo;
   ipnts[0][2]=klo;
   ipnts[1][0]=ilo;
   ipnts[1][1]=jhi;
   ipnts[1][2]=khi;
   cg_boco_write(index_file,index_base,index_zone,"Ilo",BCFarfield,PointRange,2,*ipnts,&index_bc);
   //outlet face
   ipnts[0][0]=ihi;
   ipnts[0][1]=jlo;
   ipnts[0][2]=klo;
   ipnts[1][0]=ihi;
   ipnts[1][1]=jhi;
   ipnts[1][2]=khi;
   cg_boco_write(index_file,index_base,index_zone,"Ihi",BCFarfield,PointRange,2,*ipnts,&index_bc);
   //symmetry face 1
   ipnts[0][0]=ilo;
   ipnts[0][1]=jlo;
   ipnts[0][2]=klo;
   ipnts[1][0]=ihi;
   ipnts[1][1]=jlo;
   ipnts[1][2]=khi;
   cg_boco_write(index_file,index_base,index_zone,"Jlo",BCSymmetryPlane,PointRange,2,*ipnts,&index_bc);
   //symmetry face 2
   ipnts[0][0]=ilo;
   ipnts[0][1]=jhi;
   ipnts[0][2]=klo;
   ipnts[1][0]=ihi;
   ipnts[1][1]=jhi;
   ipnts[1][2]=khi;
   cg_boco_write(index_file,index_base,index_zone,"Jhi",BCSymmetryPlane,PointRange,2,*ipnts,&index_bc);
   //wall face 1
   ipnts[0][0]=ilo;
   ipnts[0][1]=jlo;
   ipnts[0][2]=klo;
   ipnts[1][0]=ihi1;
   ipnts[1][1]=jhi;
   ipnts[1][2]=klo;
   cg_boco_write(index_file,index_base,index_zone,"Klo1",BCSymmetryPlane,PointRange,2,*ipnts,&index_bc);
   //wall face 2
   ipnts[0][0]=ihi1;
   ipnts[0][1]=jlo;
   ipnts[0][2]=klo;
   ipnts[1][0]=ihi2;
   ipnts[1][1]=jhi;
   ipnts[1][2]=klo;
   cg_boco_write(index_file,index_base,index_zone,"Klo2",BCWallViscousHeatFlux,PointRange,2,*ipnts,&index_bc);
   //wall face 3
   ipnts[0][0]=ihi2;
   ipnts[0][1]=jlo;
   ipnts[0][2]=klo;
   ipnts[1][0]=ihi;
   ipnts[1][1]=jhi;
   ipnts[1][2]=klo;
   cg_boco_write(index_file,index_base,index_zone,"Klo3",BCSymmetryPlane,PointRange,2,*ipnts,&index_bc);
   //upper face
   ipnts[0][0]=ilo;
   ipnts[0][1]=jlo;
   ipnts[0][2]=khi;
   ipnts[1][0]=ihi;
   ipnts[1][1]=jhi;
   ipnts[1][2]=khi;
   cg_boco_write(index_file,index_base,index_zone,"Khi",BCFarfield,PointRange,2,*ipnts,&index_bc);
   cg_close(index_file);
   printf("\nSuccessfully wrote BCs to file grid_c.cgns\n");
   return 0;
}

// distribute close to the center
double xDistFunc(int ind, int imax, double lX)
{
   //first and last 6th of the domain are symmetry boundary conditions
   double lXsix = lX/6.;
   double lXhalf = lX/2.;
   double lXfivesix = lX*5./6.;


   double C1 = 2.5;
   double C2 = 2.0;
   double C3 = 1.2;

   double fr1 = 1./8.;
   double fr2 = 1./6.;
   double fr3 = 3./6.;
   double fr4 = 5./6.;
   double fr5 = 7./8.;

   //first eighth of points up to leading edge
   double frac = ind/(double)imax;
   if((frac - fr1) < 0.0)
   {
      return lXsix*(1.-(1+tanh(C1*((1.-frac/fr1)-1.)/tanh(C1))));
   }
   if((frac - fr1) == 0.0)
   {
      return lXsix;
   }
   if((frac - fr1) > 0.0 && (frac - fr2) < 0.0)
   {
      double mid = fr2-fr1;
      return lXsix*(1+tanh(C2*((frac-fr1)/mid-1.))/tanh(C2)) + lXsix;
   }
   if((frac - fr2) == 0.0)
   {
      return 2*lXsix;
   }
   if((frac - fr2) > 0.0 && (frac - fr3) < 0.0)
   {
      double mid = fr3-fr2;
      return lXsix*(1.-(1+tanh(C3*(-(frac-fr2)/mid))/tanh(C3))) + 2*lXsix;

      //return (lXfivesix-lXsix)*(frac-fr1)/mid + lXsix;
   }
   if((frac - fr3) == 0.0)
   {
      return lXhalf;
   }
   if((frac - fr3) > 0.0 && (frac - fr4) < 0.0)
   {
      double mid = fr4-fr3;
      return lXsix*(1+tanh(C3*((frac-fr3)/mid-1.))/tanh(C3)) + lXhalf;
   }
   if((frac - fr4) == 0.0)
   {
      return lXhalf + lXsix;
   }
   if((frac - fr4) > 0.0 && (frac - fr5) < 0.0)
   {
      double mid = fr5-fr4;
      return lXsix*(1.-(1+tanh(C2*(-(frac-fr4)/mid)/tanh(C2)))) + lXhalf + lXsix;
   }
   if((frac - fr5) == 0.0)
   {
      return lXhalf + 2*lXsix;
   }
   if((frac - fr5) > 0.0)
   {
      double end = 1.-fr5;
      return lXsix*(1+tanh(C1*((frac-fr5)/end-1.)/tanh(C1))) + lXfivesix;
   }


}

double zDistFunc(int ind, int imax, double lX)
{
   double frac = ind/(double)imax;

   double C = 3.0;

   return lX*(1+tanh(C*(frac - 1.0))/tanh(C));
}