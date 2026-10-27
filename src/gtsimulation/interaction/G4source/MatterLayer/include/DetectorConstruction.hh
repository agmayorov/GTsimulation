#pragma once

#include <G4VUserDetectorConstruction.hh>

#include <string>
#include <vector>

class G4Tubs;
class G4LogicalVolume;
class G4Material;

namespace MatterLayer
{

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction();
    ~DetectorConstruction() override = default;
    G4VPhysicalVolume* Construct() override;

    void UpdateParameters(double thickness,
                          double density,
                          const std::vector<std::string>& elementName,
                          const std::vector<double>& elementAbundance);

    int materialCount;

  private:
    G4Tubs* fWorldSolid = nullptr;
    G4LogicalVolume* fWorldLogic = nullptr;
    G4Material* fWorldMaterial = nullptr;
    double fCurrentDensity = -1.0;
    std::vector<std::string> fCurrentNames;
    std::vector<double> fCurrentAbundances;
};

}
