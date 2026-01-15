#include "DetectorConstruction.hh"

namespace MatterLayer
{

DetectorConstruction::DetectorConstruction()
{}

DetectorConstruction::~DetectorConstruction()
{}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  fWorldSolid = new G4Tubs("World", 0.*m, 1.*m, 2.*m / 2., 0.*deg, 360.*deg);
  fWorldMaterial = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
  fWorldLogic = new G4LogicalVolume(fWorldSolid, fWorldMaterial, "World");
  G4VPhysicalVolume* worldPhys = new G4PVPlacement(
    nullptr,            // no rotation
    G4ThreeVector(),    // at (0, 0, 0)
    fWorldLogic,        // its logical volume
    "World",            // its name
    nullptr,            // its mother volume
    false,              // no boolean operation
    0,                  // copy number
    false               // overlaps checking
  );
  return worldPhys;
}

void DetectorConstruction::UpdateParameters(
  G4double thickness,
  G4double density,
  std::vector<std::string> elementName,
  std::vector<G4double> elementAbundance
) {
  fWorldSolid->SetOuterRadius(thickness * m);
  fWorldSolid->SetZHalfLength(thickness * m);
  G4RunManager::GetRunManager()->GeometryHasBeenModified();
}

}
