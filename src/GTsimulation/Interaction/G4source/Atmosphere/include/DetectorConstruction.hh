#ifndef DetectorConstruction_hh
#define DetectorConstruction_hh

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"

#include "G4SystemOfUnits.hh"
#include "G4NistManager.hh"
#include "G4Sphere.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "AtmosphereParameterisation.hh"
#include "G4PVParameterised.hh"

namespace Atmosphere
{

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction(G4double earthRadius, nrlmsise_input input);
    ~DetectorConstruction() override;
    virtual G4VPhysicalVolume *Construct() override;

  private:
    G4double fEarthRadius;
    nrlmsise_input fNrlmsiseInput;
    G4double fMaxHeight = 80.; // km
    G4double fThicknessOfOneLayer = 1.; // km
};

}

#endif
