#ifndef DetectorConstruction_hh
#define DetectorConstruction_hh

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"

#include "G4SystemOfUnits.hh"
#include "G4NistManager.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"

#include "nrlmsise-00.hh"

namespace Atmosphere
{

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction(G4double R, G4int doy, G4double sec, \
      G4double lat, G4double lon, G4double f107A, G4double f107, G4int ap);
    ~DetectorConstruction() override;
    virtual G4VPhysicalVolume *Construct() override;

  private:
    G4double fR;
    G4int fDoy;
    G4double fSec;
    G4double fLat;
    G4double fLon;
    G4double fF107A;
    G4double fF107;
    G4double fAp;
    G4double fMaxHeight = 80.; // km
    G4double fThicknessOfOneLayer = 1.; // km
};

}

#endif
