function  [rho_HI, nHI] = MW_ISM(X, Y, Z, DEN)


double Galprop::GasFunction::operator () ( const double x, const double y, const double z, const vec3 &dir ) const{
   const double r = sqrt(x*x + y*y);
   const double rScale = pow(r, frInd);
   switch (ftype) {
      case HI:
	 return nHI3D( x, y, z )*rScale;
      case CO:
	 return 2*nH23D( x, y, z, 1.0 )*rScale;
      case H2:
	 return 2*nH23D( x, y, z, fgp.fX_CO(r) )*rScale;
      case HII:
	 return nHII3D( x, y, z )*rScale;
      default:
	 return 0;
   }
}



    R = sqrt(X.^2 + Y.^2);
    theta = atan2(Y, X);