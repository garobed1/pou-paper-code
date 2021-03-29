/*   Program bump_mesh_def.c    */
/*
Creates simple 3-D structured grid and writes it to a
CGNS file.

Example compilation for this program is (change paths!):

gcc bump_mesh_def.c -L $CGNS_HOME/lib -I $CGNS_HOME/include -o bump_mesh_def

(../CGNS_CVS/cgnslib/LINUX/ is the location where the compiled
library libcgns.a is located)
*/

#include <iostream>
#include <cmath>
//#include <stdlib>
#include <string.h>
/* cgnslib.h file must be located in directory specified by -I during compile: */
#include "cgnslib.h"
#define PI 3.141592654

#if CGNS_VERSION < 3100
# define cgsize_t int
#else
# if CG_BUILD_SCOPE
#  error enumeration scoping needs to be off
# endif
#endif

double xDistFunc(int ind, int imax, double lX, double xs[][2], double *B, double *del, double *fr);

double zDistFunc(int ind, int imax, double lX, double *xs, double B, double del);

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
   ni=96;
   nj=2;
   nk=49;
   double x[nk*nj*ni],y[nk*nj*ni],z[nk*nj*ni];

/* off-wall spacings */

//z direction
   double zs[2] = {1e-4,0.1};
   double Bz = 1./((nk-1)*sqrt(zs[1]*zs[0]));
//solve transcendental equation sinh(delz)/delz = B
   int maxiter = 100;
   int count = 0;
   double drz = 0.0;
   double delz = 10.0;
   double rz = sinh(delz)/delz - Bz;
   while(abs(rz) > 1e-8 && (count - maxiter) < 0)
   {
      drz = (delz*cosh(delz)-sinh(delz))/(delz*delz);
      delz = delz - rz/drz;
      rz = sinh(delz)/delz - Bz;
      count++;
   }

//x direction, 7 sections, middle section even
   double fr[6] = {
      1./8.,
      1./6.,
      1./4.,
      3./4.,
      5./6.,
      7./8.};
   double xs[6][2] = {
   {0.1,1e-2},
   {1e-2,0.1},
   {0.1,1e-4},
   {1e-4,0.1},
   {0.1,1e-2},
   {1e-2,0.1}};
   double Bx[6] = {
   1./((fr[0]-0.)*(ni-1)*sqrt(xs[0][1]*xs[0][0])),
   1./((fr[1]-fr[0])*(ni-1)*sqrt(xs[1][1]*xs[1][0])),
   1./((fr[2]-fr[1])*(ni-1)*sqrt(xs[2][1]*xs[2][0])),
   1./((fr[4]-fr[3])*(ni-1)*sqrt(xs[3][1]*xs[3][0])),
   1./((fr[5]-fr[4])*(ni-1)*sqrt(xs[4][1]*xs[4][0])),
   1./((1.-fr[5])*(ni-1)*sqrt(xs[5][1]*xs[5][0]))};
   
   double delx[6];

   double drx = 0.;
   double dx = 0.;
   double rx = 0.;
   for(int nsec = 0; nsec < 6; nsec++)
   {  
      count = 0;
      dx = 10.;
      rx = sinh(dx)/dx - Bx[nsec];
      while(abs(rx) > 1e-8 && (count - maxiter) < 0)
      {
         drx = (dx*cosh(dx)-sinh(dx))/(dx*dx);
         dx = dx - rx/drx;
         rx = sinh(dx)/dx - Bx[nsec];
         count++;
      }
      delx[nsec] = dx;
   }
   

   for (k=0; k < nk; ++k)
   {
     for (j=0; j < nj; ++j)
     {
       for (i=0; i < ni; ++i)
       {
         x[k*nj*ni + j*ni + i]=xDistFunc(i, ni-1, lX, xs, Bx, delx, fr);//(i/((double)ni-1.))*lX;//
         y[k*nj*ni + j*ni + i]=(j/((double)nj-1.))*lY;
         z[k*nj*ni + j*ni + i]=zDistFunc(k, nk-1, lZ, zs, Bz, delz);//(k/((double)nk-1.))*lZ;
       }
     }
   }
   printf("\ncreated simple 3-D grid points");

/* WRITE X, Y, Z GRID POINTS TO CGNS FILE */
/* open CGNS file for write */
   if (cg_open("grid_c_dx_dz.cgns",CG_MODE_WRITE,&index_file)) cg_error_exit();
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
   if (cg_open("grid_c_dx_dz.cgns",CG_MODE_MODIFY,&index_file)) cg_error_exit();
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
double xDistFunc(int ind, int imax, double lX, double xs[][2], double *B, double *del, double *fr)
{
   //first and last 7 sections of the domain are symmetry boundary conditions
   double lXpre = lX/6.;
   double lXsec = lX/12.;
   double lXbum = lX/3.;

   double A, u, X;
   u = 0.;

   //first eighth of points up to leading edge
   double frac = ind/(double)imax;
   if((frac - fr[0]) < 0.0)
   {
      A = sqrt(xs[0][1]/xs[0][0]);
      u = 0.5*(1+tanh(del[0]*((frac/fr[0])-0.5))/tanh(del[0]/2.));
      X = lXpre*(u/(A+(1.-A)*u));
   }
   if((frac - fr[0]) == 0.0)
   {
      X = lXpre;
   }
   if((frac - fr[0]) > 0.0 && (frac - fr[1]) < 0.0)
   {
      double mid = fr[1]-fr[0];
      A = sqrt(xs[1][1]/xs[1][0]);
      u = 0.5*(1+tanh(del[1]*((frac-fr[0])/mid-0.5))/tanh(del[1]/2.));
      X = lXsec*(u/(A+(1.-A)*u)) + lXpre;
   }
   if((frac - fr[1]) == 0.0)
   {
      X = lXsec + lXpre;
   }
   if((frac - fr[1]) > 0.0 && (frac - fr[2]) < 0.0)
   {
      double mid = fr[2]-fr[1];
      A = sqrt(xs[2][1]/xs[2][0]);
      u = 0.5*(1+tanh(del[2]*((frac-fr[1])/mid-0.5))/tanh(del[2]/2.));
      X = lXsec*(u/(A+(1.-A)*u)) + lXsec + lXpre;

   }
   if((frac - fr[2]) == 0.0)
   {
      X = 2*lXsec + lXpre;
   }
   if((frac - fr[2]) > 0.0 && (frac - fr[3]) < 0.0)
   {
      double mid = fr[3]-fr[2];
      X = lXbum*((frac-fr[2])/mid) + 2*lXsec + lXpre;
   }
   if((frac - fr[3]) == 0.0)
   {
      X = lXbum + 2*lXsec + lXpre;
   }
   if((frac - fr[3]) > 0.0 && (frac - fr[4]) < 0.0)
   {
      double mid = fr[4]-fr[3];
      A = sqrt(xs[3][1]/xs[3][0]);
      u = 0.5*(1+tanh(del[3]*((frac-fr[3])/mid-0.5))/tanh(del[3]/2.));
      X = lXsec*(u/(A+(1.-A)*u)) + lXbum + 2*lXsec + lXpre;
   }
   if((frac - fr[4]) == 0.0)
   {
      X = lXbum + 3*lXsec + lXpre;
   }
   if((frac - fr[4]) > 0.0 && (frac - fr[5]) < 0.0)
   {
      double mid = fr[5]-fr[4];
      A = sqrt(xs[4][1]/xs[4][0]);
      u = 0.5*(1+tanh(del[4]*((frac-fr[4])/mid-0.5))/tanh(del[4]/2.));
      X = lXsec*(u/(A+(1.-A)*u)) + lXbum + 3*lXsec + lXpre;
   }
   if((frac - fr[5]) == 0.0)
   {
      X = lXbum + 4*lXsec + lXpre;
   }
   if((frac - fr[5]) > 0.0)
   {
      double end = 1.-fr[5];
      A = sqrt(xs[5][1]/xs[5][0]);
      u = 0.5*(1+tanh(del[5]*((frac-fr[5])/end-0.5))/tanh(del[5]/2.));
      X = lXpre*(u/(A+(1.-A)*u)) + lXbum + 4*lXsec + lXpre;
   }

   printf("f = %f\n", frac);
   printf("u = %f\n", u);
   printf("x = %f\n", X);

   return X;
}

double zDistFunc(int ind, int imax, double lX, double *xs, double B, double del)
{
   double frac = ind/(double)imax;

   double s1 = xs[0], s2 = xs[1];

   double A = sqrt(s2/s1);
   //double B = 1./(imax*sqrt(s2*s1));
   //double C = 3.0;

   

   double u = 0.5*(1+tanh(del*(frac - 0.5))/tanh(del/2.));
   double X = lX*(u/(A+(1.-A)*u));
   // printf("f = %f\n", frac);
   // printf("u = %f\n", u);
   // printf("x = %f\n", X);
   return X;
}