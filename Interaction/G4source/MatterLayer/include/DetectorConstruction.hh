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
      G4double w_H, G4double w_He, G4double w_N, G4double w_O, G4double w_Ar);
    ~DetectorConstruction();
    virtual G4VPhysicalVolume* Construct() override;

  private:
    G4double fThickness;
    G4double fDensity;
    G4double fW_H;
    G4double fW_He;
    G4double fW_N;
    G4double fW_O;
    G4double fW_Ar;
};

}

#endif
