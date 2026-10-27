#pragma once

#include <G4VUserDetectorConstruction.hh>
#include <G4Types.hh>
#include "nrlmsise-00.hh"

namespace Atmosphere
{

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction(G4double earthRadius, G4double maxHeight, nrlmsise_input input);
    ~DetectorConstruction() override = default;
    G4VPhysicalVolume* Construct() override;

  private:
    G4double fEarthRadius;
    G4double fMaxHeight; // m
    nrlmsise_input fNrlmsiseInput;
    G4double fSurfaceThickness = 500.; // m
    G4double fThicknessOfOneLayer = 1000.; // m
};

}
