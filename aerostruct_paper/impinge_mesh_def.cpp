/*   Program impinge_mesh_def.cpp    */
/*
Create 

compilation:

g++ impinge_mesh_def.cpp -L $CGNS_HOME/lib -lcgns -I $CGNS_HOME/include -o impinge_mesh_def


*/

#include <sstream>
#include <iostream>
#include <cmath>
//#include <stdlib>
#include <string.h>
/* cgnslib.h file must be located in directory specified by -I during compile: */
#include "cgnslib.h"
#define PI 3.141592654
#define YTL 0.26938972

#if CGNS_VERSION < 3100
# define cgsize_t int
#else
# if CG_BUILD_SCOPE
#  error enumeration scoping needs to be off
# endif
#endif

double xDistFunc(int ind, int imax, double lX, double xs, double del, double *fr);

double zDistFunc(int ind, int imax, double lX, double xs, double del);

double transcend_approx(double y);

void impinging_shock(double a_flow, double M0, double P0, double r0, double T0, 
                     double &a_shock, double &M1, double &P1, double &r1, double &T1, double &lZ);

int main(int argc, char *argv[])
{
   // parse options
   if(argc != 5)
   {
      printf("Incorrect number of arguments: Need -nx, -nz, -wall, name \n");
      return 1;
   }
   std::stringstream cx{argv[1]};   
   std::stringstream cz{argv[2]};
   std::stringstream walls{argv[3]};
   std::stringstream name{argv[4]};
   const std::string names = name.str() + "_" + cx.str() + "_" + cz.str() + "_" + walls.str() + ".cgns";
   const char* cstr = names.c_str();

   cgsize_t isize[3][3];
   cgsize_t ipnts[3][3];
   cgsize_t ids[1]; ids[0] = 1;
   int ni,nj,nk,i,j,k;
   int wall;
   int index_file,icelldim,iphysdim,index_base;
   int index_zone,index_coord;
   char basename[33],zonename[33];

/* create gridpoints: */
   double lX = 2.5;
   double lY = 1.0;
   double lZ = 0.15;
   // ni=96;
   nj=2;
   // nk=49;
   cx >> ni;
   cz >> nk;
   walls >> wall;

   double x[nk*nj*ni],y[nk*nj*ni],z[nk*nj*ni];


/* hardcode boundary settings */
   double Rs = 287.055; double gam = 1.4;
   double M0, M1, P0, P1, T0, T1, r0, r1;
   double a_shock, a_flow;
   a_shock = 25.;
   M0 = 3.;
   P0 = 2919.;
   T0 = 217.;
   r0 = P0/(Rs*T0);
   impinging_shock(a_shock, M0, P0, r0, T0, a_flow, M1, P1, r1, T1, lZ);
   double a = sqrt(gam*P0/r0); //speed of sound

   a_flow *= PI/180.;

/* off-wall spacings */

//let's just try one-sided stretching, assuming dx, dz are less than 1

//z direction
//    double zs[2] = {2e-5,0.12};
//    double Bz = 1./((nk-1)*sqrt(zs[1]*zs[0]));
// //solve transcendental equation sinh(delz)/delz = B
//    int maxiter = 1000;
//    int count = 0;
//    double drz = 0.0;
//    double delz = 10.0;
//    double rz = sinh(delz)/delz - Bz;
//    while(fabs(rz) > 1e-8 && (count - maxiter) < 0)
//    {
//       drz = (delz*cosh(delz)-sinh(delz))/(delz*delz);
//       delz = delz - rz/drz;
//       rz = sinh(delz)/delz - Bz;
//       count++;
//    }
   double zs = 2e-5;
   double dz = transcend_approx(zs);
   

//x direction, 3 sections, middle section even
   double fr[2] = {
      5./12.,
      10./12.};
   // double off1 = 3.*lX/(ni); 
   double offm = lX/((fr[1]-fr[0])*ni);
   // double off2 = 3.*lX/((1.-fr[1])*ni);
   // double xs[2][2] = {
   // {off1,offm}, 
   // {offm,off2}};
   // double Bx[2] = {
   // 1./((fr[0]-0.)*(ni-1)*sqrt(xs[0][1]*xs[0][0])),
   // //1./((fr[1]-fr[0])*(ni-1)*sqrt(xs[1][1]*xs[1][0])),
   // 1./((1.-fr[1])*(ni-1)*sqrt(xs[1][1]*xs[1][0]))};
   
   double dx = transcend_approx(offm);
   
   // double delx[2];

   // double drx = 0.;
   // double dx = 0.;
   // double rx = 0.;
   // for(int nsec = 0; nsec < 2; nsec++)
   // {  
   //    count = 0;
   //    dx = 1.;
   //    rx = sinh(dx)/dx - Bx[nsec];
   //    while(fabs(rx) > 1e-8 && (count - maxiter) < 0)
   //    {
   //       drx = (dx*cosh(dx)-sinh(dx))/(dx*dx);
   //       dx = dx - 0.1*rx/drx;
   //       rx = sinh(dx)/dx - Bx[nsec];
   //       count++;
   //    }
   //    delx[nsec] = dx;
   // }
   

   for (k=0; k < nk; ++k)
   {
     for (j=0; j < nj; ++j)
     {
       for (i=0; i < ni; ++i)
       {
         x[k*nj*ni + j*ni + i]=xDistFunc(i, ni-1, lX, offm, dx, fr);
         y[k*nj*ni + j*ni + i]=(j/((double)nj-1.))*lY;
         z[k*nj*ni + j*ni + i]=zDistFunc(k, nk-1, lZ, zs, dz);
       }
     }
   }
   printf("\ncreated simple 3-D grid points");

/* WRITE X, Y, Z GRID POINTS TO CGNS FILE */
/* open CGNS file for write */
   if (cg_open(cstr,CG_MODE_WRITE,&index_file)) cg_error_exit();
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
   printf("\nSuccessfully wrote grid to file\n");
   
   //boundary conditions
   if (cg_open(cstr,CG_MODE_MODIFY,&index_file)) cg_error_exit();
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
   double fr1 = 0.0;
   double fr5 = 1.0;
   if (wall == 2)
   {
      fr1 = fr[0];
      fr5 = fr[1];
   }
   ihi1 = (int)(fr1*ihi);
   ihi2 = (int)(fr5*ihi);
   int ier;
   int bcind;
   int dsind;
   double data[1];
   //inlet face
   ipnts[0][0]=ilo;
   ipnts[0][1]=jlo;
   ipnts[0][2]=klo;
   ipnts[1][0]=ilo;
   ipnts[1][1]=jhi;
   ipnts[1][2]=khi;
   cg_boco_write(index_file,index_base,index_zone,"Ilo",BCInflowSupersonic,PointRange,2,*ipnts,&index_bc);
   //write inlet conditions
   bcind = 1;
   dsind = 1;
   ier = cg_dataset_write(index_file,index_base,index_zone,bcind,"BCDataSet",BCInflowSupersonic,&index_bc);
   ier = cg_gopath(index_file,"/Base/Zone  1/ZoneBC/Ilo/BCDataSet/");
   ier = cg_bcdata_write(index_file,index_base,index_zone,bcind,dsind,Dirichlet);
   ier = cg_gopath(index_file,"/Base/Zone  1/ZoneBC/Ilo/BCDataSet/DirichletData");
   data[0] = P0; 
   ier = cg_array_write("Pressure",RealDouble,1,ids,data);
   data[0] = r0; 
   ier = cg_array_write("Density",RealDouble,1,ids,data);
   data[0] = M0*a; 
   ier = cg_array_write("VelocityX",RealDouble,1,ids,data);
   data[0] = 0.; 
   ier = cg_array_write("VelocityY",RealDouble,1,ids,data);
   data[0] = 0.; 
   ier = cg_array_write("VelocityZ",RealDouble,1,ids,data);


   
   //outlet face
   ipnts[0][0]=ihi;
   ipnts[0][1]=jlo;
   ipnts[0][2]=klo;
   ipnts[1][0]=ihi;
   ipnts[1][1]=jhi;
   ipnts[1][2]=khi;
   cg_boco_write(index_file,index_base,index_zone,"Ihi",BCOutflowSupersonic,PointRange,2,*ipnts,&index_bc);
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
   cg_boco_write(index_file,index_base,index_zone,"Klo1",BCWallInviscid,PointRange,2,*ipnts,&index_bc);
   //wall face 2
   ipnts[0][0]=ihi1;
   ipnts[0][1]=jlo;
   ipnts[0][2]=klo;
   ipnts[1][0]=ihi2;
   ipnts[1][1]=jhi;
   ipnts[1][2]=klo;
   cg_boco_write(index_file,index_base,index_zone,"Klo2",BCWallInviscid,PointRange,2,*ipnts,&index_bc);
   //wall face 3
   ipnts[0][0]=ihi2;
   ipnts[0][1]=jlo;
   ipnts[0][2]=klo;
   ipnts[1][0]=ihi;
   ipnts[1][1]=jhi;
   ipnts[1][2]=klo;
   cg_boco_write(index_file,index_base,index_zone,"Klo3",BCWallInviscid,PointRange,2,*ipnts,&index_bc);
   //upper face
   ipnts[0][0]=ilo;
   ipnts[0][1]=jlo;
   ipnts[0][2]=khi;
   ipnts[1][0]=ihi;
   ipnts[1][1]=jhi;
   ipnts[1][2]=khi;
   //index_bc = 8;
   cg_boco_write(index_file,index_base,index_zone,"Khi",BCInflowSupersonic,PointRange,2,*ipnts,&index_bc);
   bcind = 8;
   dsind = 1;
   ier = cg_dataset_write(index_file,index_base,index_zone,bcind,"BCDataSet",BCInflowSupersonic,&index_bc);
   ier = cg_gopath(index_file,"/Base/Zone  1/ZoneBC/Khi/BCDataSet/");
   ier = cg_bcdata_write(index_file,index_base,index_zone,bcind,dsind,Dirichlet);
   ier = cg_gopath(index_file,"/Base/Zone  1/ZoneBC/Khi/BCDataSet/DirichletData");
   // data[0] = P1; 
   // ier = cg_array_write("PressureStagnation",RealDouble,1,ids,data);
   // data[0] = r1; 
   // ier = cg_array_write("DensityStagnation",RealDouble,1,ids,data);
   // data[0] = cos(a_flow); 
   // ier = cg_array_write("VelocityUnitVectorX",RealDouble,1,ids,data);
   // data[0] = 0.; 
   // ier = cg_array_write("VelocityUnitVectorY",RealDouble,1,ids,data);
   // data[0] = -sin(a_flow); 
   // ier = cg_array_write("VelocityUnitVectorZ",RealDouble,1,ids,data);
   data[0] = P1; 
   ier = cg_array_write("Pressure",RealDouble,1,ids,data);
   data[0] = r1; 
   ier = cg_array_write("Density",RealDouble,1,ids,data);
   data[0] = cos(a_flow)*M1*a; 
   ier = cg_array_write("VelocityX",RealDouble,1,ids,data);
   data[0] = 0.; 
   ier = cg_array_write("VelocityY",RealDouble,1,ids,data);
   data[0] = -sin(a_flow)*M1*a; 
   ier = cg_array_write("VelocityZ",RealDouble,1,ids,data);
   cg_close(index_file);
   printf("\nSuccessfully wrote BCs to file\n");
   return 0;
}

// distribute close to the center
double xDistFunc(int ind, int imax, double lX, double xs, double del, double *fr)
{
   // just do three sections
   double lXpre = lX*4./5.;
   double lXbum = lX*25.4/250.; // in centimeters
   double lXpos = lX-lXbum-lXpre;

   double t, X;
   //points up to leading edge
   double frac = ind/(double)imax;
   if((frac - fr[0]) < 0.0)
   {
      // A = sqrt(xs[0][1]/xs[0][0]);
      // u = 0.5*(1+tanh(del[0]*((frac/fr[0])-0.5))/tanh(del[0]/2.));
      // X = lXpre*(u/(A+(1.-A)*u));
      // double t = 1. + tan(del*(frac - 1.))/tan(del);
      t = (1./del)*asinh((frac/fr[0])*sinh(del));
      X = lXpre*t;
   }
   if((frac - fr[0]) == 0.0)
   {
      X = lXpre;
   }
   if((frac - fr[0]) > 0.0 && (frac - fr[1]) < 0.0)
   {
      double mid = fr[1]-fr[0];
      X = lXbum*(frac - fr[0])/mid + lXpre;
   }
   if((frac - fr[1]) == 0.0)
   {
      X = lXbum + lXpre;
   }
   if((frac - fr[1]) > 0.0)
   {
      double mid = 1.0-fr[1];
      // A = 1.0;
      // u = 0.5*(1+tanh(del[1]*((frac-fr[1])/mid-0.5))/tanh(del[1]/2.));
      // X = lXpos*(u/(A+(1.-A)*u)) + lXbum + lXpre;
      //X = lXbum*((frac-fr[2])/mid) + 2*lXsec + lXpre;

      //double t = 1. + tan(del*((frac-fr[1])/mid - 1.))/tan(del);
      t = sinh(del*((frac-fr[1])/mid))/sinh(del);
      X = lXpos*t + lXbum + lXpre;
   }

   // printf("f = %f\n", frac);
   // printf("u = %f\n", u);
   // printf("x = %f\n", X);

   //X = frac*(lXpre+lXbum+lXpos);
   return X;
}

double zDistFunc(int ind, int imax, double lX, double xs, double del)
{
   double frac = ind/(double)imax;

   //double s1 = xs[0], s2 = xs[1];

   //double A = sqrt(s2/s1);
   //double B = 1./(imax*sqrt(s2*s1));
   //double C = 3.0;

   double t = sinh(del*(frac))/sinh(del);
   double X = lX*t;
   

   // double u = 0.5*(1+tanh(del*(frac - 0.5))/tanh(del/2.));
   // double X = lX*(u/(A+(1.-A)*u));
   // printf("f = %f\n", frac);
   // printf("u = %f\n", u);
   // printf("x = %f\n", X);
   return X;
}

double transcend_approx(double y)
{
   //solve transcendental equation sinh(2x)/2x = y
   //use approximate analytic solution (from Vinokur 1983)
   double x;
   if (y < YTL)
   {
      x = PI*(1. - y + y*y - (1. + PI*PI/6.)*y*y*y + 
         6.794732*y*y*y*y - 13.205501*y*y*y*y*y + 11.726095*y*y*y*y*y*y);
   }
   else
   {
      double yb = 1. - y;
      x = sqrt(6*yb)*(1. + 0.15*yb + 0.057321429*yb*yb + 0.048774238*yb*yb*yb -
         0.053337753*yb*yb*yb*yb - 0.075845134*yb*yb*yb*yb*yb);
   }
   //x /= 2.;
   return x;
}

void impinging_shock(double a_shock, double M0, double P0, double r0, double T0, 
                     double &a_flow, double &M1, double &P1, double &r1, double &T1, double &lZ)
{
   double s = a_shock*PI/180.;
   double ss = sin(s);
   double gam = 1.4;
   double work;

   //determine lZ that puts shock in top left corner
   lZ = 2.254*tan(s);

   //determine flow angle
   work = (gam + 1)*M0*M0/(2*(M0*M0*ss*ss - 1)) - 1;
   work *= tan(s);
   double a = atan(1./work);
   a_flow = a*180./PI;

   //determine downstream mach number
   work = ((gam-1)*M0*M0*ss*ss + 2)/(2*gam*M0*M0*ss*ss - (gam - 1));
   work /= (sin(s-a)*sin(s-a));
   M1 = sqrt(work);

   //downstream pressure
   work = P0*(2*gam*M0*M0*ss*ss - (gam - 1));
   work /= (gam + 1);
   P1 = work;

   //downstream density
   work = r0*(gam + 1)*M0*M0*ss*ss;
   work /= ((gam-1)*M0*M0*ss*ss + 2);
   r1 = work;

   //downstream temperature (in case)
   work = T0*(2*gam*M0*M0*ss*ss - (gam - 1))*((gam-1)*M0*M0*ss*ss + 2);
   work /= ((gam+1)*(gam+1)*M0*M0*ss*ss);
   T1 = work;

}