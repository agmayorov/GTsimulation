#include "AtmosphereParameterisation.hh"

#include <G4Sphere.hh>
#include <G4VPhysicalVolume.hh>
#include <G4ThreeVector.hh>
#include <G4SystemOfUnits.hh>

namespace Atmosphere
{

AtmosphereParameterisation::AtmosphereParameterisation(G4double thicknessOfLayer, G4double earthRadius, std::vector<G4Material*> atmospheric_layer_mat)
: fThicknessOfLayer(thicknessOfLayer),
  fEarthRadius(earthRadius),
  fAtmosphericLayerMat(atmospheric_layer_mat)
{}

void AtmosphereParameterisation::ComputeTransformation(const G4int copyNo, G4VPhysicalVolume* physVol) const
{
  physVol->SetTranslation(G4ThreeVector());
  physVol->SetRotation(nullptr);
}

void AtmosphereParameterisation::ComputeDimensions(G4Sphere& atmosphericLayer, const G4int copyNo, const G4VPhysicalVolume* physVol) const
{
  G4double innerRadius = fEarthRadius + copyNo * fThicknessOfLayer;
  G4double outerRadius = innerRadius + fThicknessOfLayer;
  atmosphericLayer.SetInnerRadius(innerRadius*km);
  atmosphericLayer.SetOuterRadius(outerRadius*km);
}

G4Material* AtmosphereParameterisation::ComputeMaterial(const G4int copyNo, G4VPhysicalVolume* /*currentVol*/, const G4VTouchable* parentTouch)
{
  return fAtmosphericLayerMat[copyNo];
}

}
