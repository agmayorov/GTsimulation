#include "DetectorConstruction.hh"

namespace Atmosphere
{

DetectorConstruction::DetectorConstruction(G4double earthRadius, nrlmsise_input input)
: G4VUserDetectorConstruction(),
  fEarthRadius(earthRadius),
  fNrlmsiseInput(input)
{}

DetectorConstruction::~DetectorConstruction()
{}

G4VPhysicalVolume *DetectorConstruction::Construct()
{
  G4NistManager *nist = G4NistManager::Instance();
  G4bool check_overlaps = false;

  //------ World -----------------------------------------------------------------------------------

  G4double innerRadius = (fEarthRadius - 0.5)*km;
  G4double outerRadius = (fEarthRadius + fMaxHeight)*km;
  G4Sphere *world_svol = new G4Sphere("world", innerRadius, outerRadius, 0., 360*deg, 0., 180*deg);

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

  innerRadius = (fEarthRadius - 0.5)*km;
  outerRadius = fEarthRadius*km;
  G4Sphere *surface_svol = new G4Sphere("surface", innerRadius, outerRadius, 0., 360*deg, 0., 180*deg);

  G4Material *surface_mat = nist->FindOrBuildMaterial("G4_SILICON_DIOXIDE");
  G4LogicalVolume *surface_lvol = new G4LogicalVolume(surface_svol, surface_mat, "surface");

  new G4PVPlacement(nullptr,            // no rotation
                    G4ThreeVector(),    // position
                    surface_lvol,       // logical volume
                    "surface",          // name
                    world_lvol,         // mother volume
                    false,              // no boolean operation
                    0,                  // copy number
                    check_overlaps);    // overlaps checking

  //------ Atmosphere ------------------------------------------------------------------------------

  innerRadius = fEarthRadius*km;
  outerRadius = (fEarthRadius + fMaxHeight)*km;
  G4Sphere *atmopshere_svol = new G4Sphere("atmosphere", innerRadius, outerRadius, 0., 360*deg, 0., 180*deg);

  G4Material *atmopshere_mat = nist->FindOrBuildMaterial("G4_AIR");
  G4LogicalVolume *atmopshere_lvol = new G4LogicalVolume(atmopshere_svol, atmopshere_mat, "atmosphere");

  new G4PVPlacement(nullptr,            // no rotation
                    G4ThreeVector(),    // position
                    atmopshere_lvol,    // logical volume
                    "atmosphere",       // name
                    world_lvol,         // mother volume
                    false,              // no boolean operation
                    0,                  // copy number
                    check_overlaps);    // overlaps checking

  //------ Atmospheric Layers ----------------------------------------------------------------------

  nrlmsise_output output;
  nrlmsise_flags flags;
  flags.switches[0] = 0;
  for (G4int i = 1; i < 24; i++)
    flags.switches[i] = 1;
  gtd7(&fNrlmsiseInput, &flags, &output);

  G4Element *He = new G4Element("Helium",   "He",  2., 4.003*g/mole);
  G4Element *N  = new G4Element("Nitrogen", "N",   7., 14.01*g/mole);
  G4Element *O  = new G4Element("Oxygen",   "O",   8., 16.00*g/mole);
  G4Element *Ar = new G4Element("Argon",    "Ar", 18., 39.95*g/mole);

  // Initialization of variables for numerical integration
  G4double layer_density = 0.;
  G4double h = 0; // km above ground
  G4int n_integrating = 100; // free parameter
  G4double dx = fThicknessOfOneLayer / n_integrating;
  G4double x2 = 0., x3 = 0., f1 = 0., f2 = 0., f3 = output.d[5];

  G4int numberOfLayers = fMaxHeight / fThicknessOfOneLayer;

  std::vector<G4Material*> atmospheric_layer_mat(numberOfLayers);

  for (G4int i = 0; i < numberOfLayers; i++) {
    layer_density = 0.;
    h = i * fThicknessOfOneLayer;
    for (G4int k = 0; k < n_integrating; k++) {
      x2 = h + dx / 2. + dx * k;
      x3 = x2 + dx / 2.;
      f1 = f3;
      fNrlmsiseInput.alt = x2;
      gtd7(&fNrlmsiseInput, &flags, &output);
      f2 = output.d[5];
      fNrlmsiseInput.alt = x3;
      gtd7(&fNrlmsiseInput, &flags, &output);
      f3 = output.d[5];
      layer_density += (f1 + 4. * f2 + f3) / 6.;
    }
    layer_density *= dx;

    std::string copyNoString = std::to_string(i);
    atmospheric_layer_mat[i] = new G4Material("Air_" + copyNoString, layer_density*g/cm3, 4);
    atmospheric_layer_mat[i]->AddElement(He, 0.00052418*perCent);
    atmospheric_layer_mat[i]->AddElement(O, 78.11046347*perCent);
    atmospheric_layer_mat[i]->AddElement(N, 20.95469404*perCent);
    atmospheric_layer_mat[i]->AddElement(Ar, 0.93431831*perCent);
  }

  outerRadius = (fEarthRadius + fThicknessOfOneLayer)*km;
  G4Sphere *atmopshere_layer_svol = new G4Sphere("atmospheric_layer", innerRadius, outerRadius, 0., 360.*deg, 0., 180.*deg);
  G4LogicalVolume *atmopshere_layer_lvol = new G4LogicalVolume(atmopshere_layer_svol, atmopshere_mat, "atmospheric_layer");

  AtmosphereParameterisation *atmosphere_param = new AtmosphereParameterisation(fThicknessOfOneLayer, fEarthRadius, atmospheric_layer_mat);
  new G4PVParameterised("atmosphere",             // name
                        atmopshere_layer_lvol,    // logical volume
                        atmopshere_lvol,          // mother volume
                        kUndefined,               // along this axis
                        numberOfLayers,           // copy number
                        atmosphere_param,         // parametrisation
                        check_overlaps);          // overlaps checking

  return world_pvol;
}

}
