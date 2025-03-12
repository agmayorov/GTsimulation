#ifndef DetectorConstruction_hh
#define DetectorConstruction_hh

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"

#include "G4SystemOfUnits.hh"
#include "G4NistManager.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"

namespace MatterLayer
{

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction(G4double thickness, G4double density, \
      std::vector<G4String> elementName, std::vector<G4double> elementAbundance);
    ~DetectorConstruction();
    virtual G4VPhysicalVolume *Construct() override;

  private:
    G4double fThickness;
    G4double fDensity;
    std::vector<G4String> fElementName;
    std::vector<G4double> fElementAbundance;
};

}

#endif
