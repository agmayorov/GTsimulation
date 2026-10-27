#include "DetectorConstruction.hh"

#include <G4SystemOfUnits.hh>
#include <G4NistManager.hh>
#include <G4Material.hh>
#include <G4Element.hh>
#include <G4Tubs.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>
#include <G4RunManager.hh>

#include <cmath>
#include <stdexcept>
#include <string>

namespace MatterLayer
{

DetectorConstruction::DetectorConstruction()
: materialCount(0)
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
  double thickness,
  double density,
  const std::vector<std::string>& elementName,
  const std::vector<double>& elementAbundance
) {
  fWorldSolid->SetOuterRadius(thickness * m);
  fWorldSolid->SetZHalfLength(thickness * m);

  bool materialParamsChanged = (
    (std::fabs(fCurrentDensity - density) > 1e-7) ||
    (fCurrentNames != elementName) ||
    (fCurrentAbundances != elementAbundance)
  );
  if (!materialParamsChanged) {
    G4RunManager::GetRunManager()->GeometryHasBeenModified();
    return;
  }

  fCurrentDensity = density;
  fCurrentNames = elementName;
  fCurrentAbundances = elementAbundance;

  std::string uniqueName = "Matter_" + std::to_string(materialCount++);
  fWorldMaterial = new G4Material(uniqueName, density * g/cm3, elementName.size());
  for (G4int i = 0; i < elementName.size(); ++i) {
    G4Element* e = G4NistManager::Instance()->FindOrBuildElement(elementName[i]);
    if (!e)
      throw std::invalid_argument("Element " + elementName[i] + " not found");
    fWorldMaterial->AddElement(e, elementAbundance[i]);
  }
  fWorldLogic->SetMaterial(fWorldMaterial);
  G4RunManager::GetRunManager()->GeometryHasBeenModified();
}

}
