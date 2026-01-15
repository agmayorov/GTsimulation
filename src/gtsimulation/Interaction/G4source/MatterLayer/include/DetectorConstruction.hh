#pragma once

#include <G4VUserDetectorConstruction.hh>
#include <G4VPhysicalVolume.hh>

#include <G4SystemOfUnits.hh>
#include <G4NistManager.hh>
#include <G4Tubs.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>
#include <G4RunManager.hh>

namespace MatterLayer
{

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction();
    ~DetectorConstruction();
    G4VPhysicalVolume* Construct() override;

    void UpdateParameters(G4double thickness,
                          G4double density,
                          std::vector<std::string> elementName,
                          std::vector<G4double> elementAbundance);

  private:
    G4Tubs* fWorldSolid = nullptr;
    G4LogicalVolume* fWorldLogic = nullptr;
    G4Material* fWorldMaterial = nullptr;
};

}
