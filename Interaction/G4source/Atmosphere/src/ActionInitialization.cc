#include "ActionInitialization.hh"

namespace Atmosphere
{

ActionInitialization::ActionInitialization(G4int particlePDG, G4double energy, G4ThreeVector coordinates, G4ThreeVector velocity)
: fParticlePDG(particlePDG),
  fEnergy(energy),
  fCoordinates(coordinates),
  fVelocity(velocity)
{}

ActionInitialization::~ActionInitialization()
{}

void ActionInitialization::Build() const
{
  SetUserAction(new PrimaryGeneratorAction(fParticlePDG, fEnergy, fCoordinates, fVelocity));
  SetUserAction(new TrackingAction());
}

}
