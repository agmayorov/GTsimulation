#pragma once

#include <G4VUserDetectorConstruction.hh>
#include <G4Types.hh>
#include "nrlmsise-00.hh"

namespace Atmosphere
{

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction(G4double earthRadius, nrlmsise_input input);
    ~DetectorConstruction() override = default;
    G4VPhysicalVolume* Construct() override;

  private:
    G4double fEarthRadius;
    nrlmsise_input fNrlmsiseInput;
    G4double fMaxHeight = 80.; // km
    G4double fSurfaceThickness = 0.5; // km
    G4double fThicknessOfOneLayer = 1.; // km
};

}
