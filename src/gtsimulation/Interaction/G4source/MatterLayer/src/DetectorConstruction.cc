#include "DetectorConstruction.hh"

namespace MatterLayer
{

DetectorConstruction::DetectorConstruction(G4double thickness, G4double density, \
  std::vector<G4String> elementName, std::vector<G4double> elementAbundance)
: G4VUserDetectorConstruction(),
  fThickness(thickness),
  fDensity(density),
  fElementName(elementName),
  fElementAbundance(elementAbundance)
{}

DetectorConstruction::~DetectorConstruction()
{}

G4VPhysicalVolume *DetectorConstruction::Construct()
{
  G4NistManager *nist = G4NistManager::Instance();
  G4double density = 0.;
  G4int nelem = 0;

  //------ World -----------------------------------------------------------------------------------

  G4double world_sizeR =    fThickness*m;
  G4double world_sizeZ = 2.*fThickness*m;
  G4Tubs *world_svol = new G4Tubs("World", 0.*m, world_sizeR, world_sizeZ/2., 0.*deg, 360.*deg);

  G4Material *world_mat = new G4Material("Matter", density = fDensity*g/cm3, nelem = fElementName.size());
  for (G4int i = 0; i < fElementName.size(); i++) {
    G4Element *e = nist->FindOrBuildElement(fElementName[i]);
    if (!e) {
      G4cerr << "Element "<< fElementName[i] <<" was not found in Geant4 Material Database" << G4endl;
      exit(3);
    }
    world_mat->AddElement(e, fElementAbundance[i]);
  }
  G4LogicalVolume *world_lvol = new G4LogicalVolume(world_svol, world_mat, "World");

  G4VPhysicalVolume *world_pvol = new G4PVPlacement(
    nullptr,            // no rotation
    G4ThreeVector(),    // at (0, 0, 0)
    world_lvol,         // its logical volume
    "World",            // its name
    nullptr,            // its mother volume
    false,              // no boolean operation
    0,                  // copy number
    false               // overlaps checking
  );

  return world_pvol;
}

}
