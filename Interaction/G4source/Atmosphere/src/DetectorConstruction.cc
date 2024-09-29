#include "DetectorConstruction.hh"

namespace Atmosphere
{

DetectorConstruction::DetectorConstruction(G4double R, G4int doy, G4double sec, \
  G4double lat, G4double lon, G4double f107A, G4double f107, G4int ap)
: G4VUserDetectorConstruction(),
  fR(R),
  fDoy(doy),
  fSec(sec),
  fLat(lat),
  fLon(lon),
  fF107A(f107A),
  fF107(f107),
  fAp(ap)
{}

DetectorConstruction::~DetectorConstruction()
{}

G4VPhysicalVolume *DetectorConstruction::Construct()
{
  G4NistManager *nist = G4NistManager::Instance();
  G4bool check_overlaps = false;

  G4double z = 0.; // atomic number
  G4double a = 0.; // atomic mass
  G4double density = 0.;
  G4int n_elem = 0;

  G4Element *He = new G4Element("Helium",   "He", z =  2., a = 4.003*g/mole);
  G4Element *N  = new G4Element("Nitrogen", "N",  z =  7., a = 14.01*g/mole);
  G4Element *O  = new G4Element("Oxygen",   "O",  z =  8., a = 16.00*g/mole);
  G4Element *Ar = new G4Element("Argon",    "Ar", z = 18., a = 39.95*g/mole);

  //------ World -----------------------------------------------------------------------------------

  G4double sizeR = (1. + fR)*km;
  G4double sizeZ = (1. + 2.*fMaxHeight)*km;
  G4Tubs *world_svol = new G4Tubs("world", 0*km, sizeR, sizeZ/2, 0*deg, 360*deg);

  G4Material *world_mat = nist->FindOrBuildMaterial("G4_Galactic");
  G4LogicalVolume *world_lvol = new G4LogicalVolume(world_svol, world_mat, "world");

  G4VPhysicalVolume *world_pvol = 
  new G4PVPlacement(nullptr,            // no rotation
                    G4ThreeVector(),    // position
                    world_lvol,         // logical volume
                    "world",            // name
                    nullptr,            // mother volume
                    false,              // no boolean operation
                    0,                  // copy number
                    check_overlaps);    // overlaps checking

  //------ Surface ---------------------------------------------------------------------------------

  sizeR = fR*km;
  sizeZ = 2.*km;
  G4Tubs *surface_svol = new G4Tubs("surface", 0*km, sizeR, sizeZ/2, 0*deg, 360*deg);

  G4Material *surface_mat = nist->FindOrBuildMaterial("G4_Si");
  G4LogicalVolume *surface_lvol = new G4LogicalVolume(surface_svol, surface_mat, "surface");

  new G4PVPlacement(nullptr,                      // no rotation
                    G4ThreeVector(0.,0.,-1.*km),  // position
                    surface_lvol,                 // logical volume
                    "surface",                    // name
                    world_lvol,                   // mother volume
                    false,                        // no boolean operation
                    0,                            // copy number
                    check_overlaps);              // overlaps checking

  //------ Atmosphere ------------------------------------------------------------------------------

  struct nrlmsise_output output;
  struct nrlmsise_input input;
  struct nrlmsise_flags flags;

  flags.switches[0] = 0;
  for (G4int i = 1; i < 24; i++)
    flags.switches[i] = 1;

  input.doy = fDoy;
  input.year = 0; /* without effect */
  input.sec = fSec;
  input.alt = 0;
  input.g_lat = fLat;
  input.g_long = fLon;
  input.lst = fSec/3600. + fLon/15.;
  input.f107A = 150.;
  input.f107 = 150.;
  input.ap = 4.;
  gtd7(&input, &flags, &output);

  // Initialization of variables for numerical integration
  G4double h = 0.;
  G4int n_integrating = 100;
  G4double dx = fThicknessOfOneLayer / n_integrating;
  G4double x2;
  G4double x3;
  G4double f1;
  G4double f2;
  G4double f3 = output.d[5];

  G4int numberOfLayers = fMaxHeight / fThicknessOfOneLayer;

  std::vector<G4Material*> atmospheric_layer_mat(numberOfLayers);
  std::vector<G4Tubs*> atmospheric_layer_solid(numberOfLayers);
  std::vector<G4LogicalVolume*> atmospheric_layer_log(numberOfLayers);

  std::vector<G4double> d_alt(numberOfLayers);
  std::vector<G4double> d_rho(numberOfLayers);

  for (G4int i = 0; i < numberOfLayers; i++)
  {
    std::string n = std::to_string(i+1);
    std::string atmospheric_layer_name = "AtmosphericLayer_" + n;

    d_alt[i] = fThicknessOfOneLayer;

    for (G4int k = 0; k < n_integrating; k++)
    {
      x2 = h + dx/2. + dx*k;
      x3 = x2 + dx/2.;
      f1 = f3;
      input.alt = x2;
      gtd7(&input, &flags, &output);
      f2 = output.d[5];
      input.alt = x3;
      gtd7(&input, &flags, &output);
      f3 = output.d[5];
      d_rho[i] = d_rho[i] + (f1 + 4.*f2 + f3)/6.;
    }
    d_rho[i] = d_rho[i]*dx;

    atmospheric_layer_mat[i] = new G4Material("Air_" + n, density = d_rho[i]*g/cm3, n_elem = 4);
    atmospheric_layer_mat[i]->AddElement(He, 0.00052418*perCent);
    atmospheric_layer_mat[i]->AddElement(O, 78.11046347*perCent);
    atmospheric_layer_mat[i]->AddElement(N, 20.95469404*perCent);
    atmospheric_layer_mat[i]->AddElement(Ar, 0.93431831*perCent);

    G4double atmospheric_layer_sizeR = fR*km;
    G4double atmospheric_layer_sizeZ = d_alt[i]*km;

    atmospheric_layer_solid[i] =
      new G4Tubs(atmospheric_layer_name,
                 0.*km,                         // inner radius
                 atmospheric_layer_sizeR,       // outer radius
                 0.5*atmospheric_layer_sizeZ,   // hight
                 0.*deg,                        // start angle
                 360.*deg);                     // spanning angle

    atmospheric_layer_log[i] =
      new G4LogicalVolume(atmospheric_layer_solid[i], // its solid
                          atmospheric_layer_mat[i],   // its material
                          atmospheric_layer_name);    // its name

    new G4PVPlacement(0,                                        // no rotation
                      G4ThreeVector(0.,0.,(h+d_alt[i]/2)*km),   // at (0,0,0.5*h)
                      atmospheric_layer_log[i],                 // its logical volume
                      atmospheric_layer_name,                   // its name
                      world_lvol,                                // its mother  volume
                      false,                                    // no boolean operation
                      0,                                        // copy number
                      check_overlaps);                           // overlaps checking

    h = h + d_alt[i];

    // if (h >= 80.) {
    //   input.f107A = fF107A;
    //   input.f107 = fF107;
    //   input.ap = fAp;
    // }
  }

  return world_pvol;
}

}
