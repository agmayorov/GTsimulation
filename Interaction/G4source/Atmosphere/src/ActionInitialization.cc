#include "ActionInitialization.hh"

namespace Atmosphere
{

ActionInitialization::ActionInitialization(G4int particlePDG, G4double energy, G4double height, G4double alpha)
: fParticlePDG(particlePDG),
  fEnergy(energy),
  fHeight(height),
  fAlpha(alpha)
{}

ActionInitialization::~ActionInitialization()
{}

void ActionInitialization::Build() const
{
  SetUserAction(new PrimaryGeneratorAction(fParticlePDG, fEnergy, fHeight, fAlpha));
  SetUserAction(new TrackingAction());
}

}
