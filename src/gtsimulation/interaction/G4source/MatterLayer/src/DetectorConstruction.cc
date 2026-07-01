#include "DetectorConstruction.hh"

namespace MatterLayer
{

DetectorConstruction::DetectorConstruction()
: fMatCounter(0)
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

  std::string uniqueName = "Matter_" + std::to_string(fMatCounter++);
  fWorldMaterial = new G4Material(uniqueName, density * g/cm3, elementName.size());
  for (G4int i = 0; i < elementName.size(); ++i) {
    G4Element* e = G4NistManager::Instance()->FindOrBuildElement(elementName[i]);
    if (!e) {
      std::cerr << "Error: Element " << elementName[i] << " not found!" << std::endl;
      exit(3);
    }
    fWorldMaterial->AddElement(e, elementAbundance[i]);
  }
  fWorldLogic->SetMaterial(fWorldMaterial);
  G4RunManager::GetRunManager()->GeometryHasBeenModified();
}

}
