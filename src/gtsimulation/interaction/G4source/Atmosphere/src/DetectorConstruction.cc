#include "DetectorConstruction.hh"
#include "AtmosphereParameterisation.hh"

#include <G4SystemOfUnits.hh>
#include <G4PhysicalConstants.hh>
#include <G4NistManager.hh>
#include <G4Material.hh>
#include <G4Element.hh>
#include <G4Sphere.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>
#include <G4PVParameterised.hh>
#include <G4ThreeVector.hh>
#include <G4String.hh>

#include <array>
#include <string>
#include <vector>

namespace Atmosphere
{

DetectorConstruction::DetectorConstruction(G4double earthRadius, nrlmsise_input input)
: G4VUserDetectorConstruction(),
  fEarthRadius(earthRadius),
  fNrlmsiseInput(input)
{}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  G4NistManager* nist = G4NistManager::Instance();
  G4bool check_overlaps = false;

  //------ World -----------------------------------------------------------------------------------

  G4double innerRadius = (fEarthRadius - fSurfaceThickness)*km;
  G4double outerRadius = (fEarthRadius + fMaxHeight)*km;
  G4Sphere* world_svol = new G4Sphere("world", innerRadius, outerRadius, 0., 360.*deg, 0., 180.*deg);

  G4Material* world_mat = nist->FindOrBuildMaterial("G4_Galactic");
  G4LogicalVolume* world_lvol = new G4LogicalVolume(world_svol, world_mat, "world");

  G4VPhysicalVolume* world_pvol =
  new G4PVPlacement(nullptr,            // no rotation
                    G4ThreeVector(),    // position
                    world_lvol,         // logical volume
                    "world",            // name
                    nullptr,            // mother volume
                    false,              // no boolean operation
                    0,                  // copy number
                    check_overlaps);    // overlaps checking

  //------ Surface ---------------------------------------------------------------------------------

  innerRadius = (fEarthRadius - fSurfaceThickness)*km;
  outerRadius = fEarthRadius*km;
  G4Sphere* surface_svol = new G4Sphere("surface", innerRadius, outerRadius, 0., 360.*deg, 0., 180.*deg);

  G4Material* surface_mat = nist->FindOrBuildMaterial("G4_SILICON_DIOXIDE");
  G4LogicalVolume* surface_lvol = new G4LogicalVolume(surface_svol, surface_mat, "surface");

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
  G4Sphere* atmosphere_svol = new G4Sphere("atmosphere", innerRadius, outerRadius, 0., 360.*deg, 0., 180.*deg);

  G4Material* atmosphere_mat = nist->FindOrBuildMaterial("G4_AIR");
  G4LogicalVolume* atmosphere_lvol = new G4LogicalVolume(atmosphere_svol, atmosphere_mat, "atmosphere");

  new G4PVPlacement(nullptr,            // no rotation
                    G4ThreeVector(),    // position
                    atmosphere_lvol,    // logical volume
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

  // Elements of the atmosphere; mass of one atom is in Geant4 internal units
  constexpr G4int n_elements = 5;
  std::array<G4String, n_elements> element_symbols = {"H", "He", "N", "O", "Ar"};
  std::array<G4Element*, n_elements> elements;
  std::array<G4double, n_elements> atom_mass;
  for (G4int e = 0; e < n_elements; e++) {
    elements[e] = nist->FindOrBuildElement(element_symbols[e]);
    atom_mass[e] = elements[e]->GetA() / Avogadro;
  }

  // Mass density of each element at given altitude [km] from NRLMSISE-00 number densities [cm-3].
  // Anomalous oxygen (d[8]) is excluded, as in the total mass density d[5] of gtd7
  auto compute_element_density = [&](G4double altitude, std::array<G4double, n_elements>& density) {
    fNrlmsiseInput.alt = altitude;
    gtd7(&fNrlmsiseInput, &flags, &output);
    density[0] = output.d[6] / cm3 * atom_mass[0];                           // H
    density[1] = output.d[0] / cm3 * atom_mass[1];                           // He
    density[2] = (2. * output.d[2] + output.d[7]) / cm3 * atom_mass[2];      // N2, N
    density[3] = (output.d[1] + 2. * output.d[3]) / cm3 * atom_mass[3];      // O, O2
    density[4] = output.d[4] / cm3 * atom_mass[4];                           // Ar
  };

  // Initialization of variables for numerical integration
  G4double h = 0; // km above ground
  G4int n_integrating = 100; // free parameter
  G4double dx = fThicknessOfOneLayer / n_integrating;
  G4double x2 = 0., x3 = 0.;
  std::array<G4double, n_elements> f1{}, f2{}, f3{};
  compute_element_density(0., f3);

  G4int numberOfLayers = fMaxHeight / fThicknessOfOneLayer;

  std::vector<G4Material*> atmospheric_layer_mat(numberOfLayers);

  for (G4int i = 0; i < numberOfLayers; i++) {
    // Integral of the element mass density over the layer height (Simpson's rule)
    std::array<G4double, n_elements> layer_mass{};
    h = i * fThicknessOfOneLayer;
    for (G4int k = 0; k < n_integrating; k++) {
      x2 = h + dx / 2. + dx * k;
      x3 = x2 + dx / 2.;
      f1 = f3;
      compute_element_density(x2, f2);
      compute_element_density(x3, f3);
      for (G4int e = 0; e < n_elements; e++)
        layer_mass[e] += (f1[e] + 4. * f2[e] + f3[e]) / 6. * dx;
    }

    G4double total_mass = 0.;
    G4int n_components = 0;
    for (G4int e = 0; e < n_elements; e++) {
      total_mass += layer_mass[e];
      if (layer_mass[e] > 0.)
        n_components++;
    }
    G4double layer_density = total_mass / fThicknessOfOneLayer;

    std::string copyNoString = std::to_string(i);
    atmospheric_layer_mat[i] = new G4Material("Air_" + copyNoString, layer_density, n_components);
    for (G4int e = 0; e < n_elements; e++) {
      if (layer_mass[e] > 0.)
        atmospheric_layer_mat[i]->AddElement(elements[e], layer_mass[e] / total_mass);
    }
  }

  outerRadius = (fEarthRadius + fThicknessOfOneLayer)*km;
  G4Sphere* atmosphere_layer_svol = new G4Sphere("atmospheric_layer", innerRadius, outerRadius, 0., 360.*deg, 0., 180.*deg);
  G4LogicalVolume* atmosphere_layer_lvol = new G4LogicalVolume(atmosphere_layer_svol, atmosphere_mat, "atmospheric_layer");

  AtmosphereParameterisation* atmosphere_param = new AtmosphereParameterisation(fThicknessOfOneLayer, fEarthRadius, atmospheric_layer_mat);
  new G4PVParameterised("atmosphere",             // name
                        atmosphere_layer_lvol,    // logical volume
                        atmosphere_lvol,          // mother volume
                        kUndefined,               // along this axis
                        numberOfLayers,           // copy number
                        atmosphere_param,         // parametrisation
                        check_overlaps);          // overlaps checking

  return world_pvol;
}

}
