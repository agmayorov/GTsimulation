#pragma once

#include <G4VPVParameterisation.hh>
#include <G4Types.hh>

#include <vector>

class G4Sphere;
class G4Material;
class G4VPhysicalVolume;

namespace Atmosphere
{

class AtmosphereParameterisation : public G4VPVParameterisation
{
  public:
    AtmosphereParameterisation(G4double thicknessOfLayer, G4double earthRadius, std::vector<G4Material*> atmospheric_layer_mat);
    ~AtmosphereParameterisation() override = default;

    void ComputeTransformation(const G4int copyNo, G4VPhysicalVolume* physVol) const override;
    void ComputeDimensions(G4Sphere& atmosphericLayer, const G4int copyNo, const G4VPhysicalVolume* physVol) const override;
    G4Material* ComputeMaterial(const G4int copyNo, G4VPhysicalVolume* currentVol, const G4VTouchable* parentTouch = nullptr) override;

  private:
    G4double fThicknessOfLayer;
    G4double fEarthRadius;
    std::vector<G4Material*> fAtmosphericLayerMat;
};

}
