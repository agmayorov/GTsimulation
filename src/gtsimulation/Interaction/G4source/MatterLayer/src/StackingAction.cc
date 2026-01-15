#include "StackingAction.hh"

namespace MatterLayer
{

StackingAction::StackingAction()
: fDeathTime(-1.)
{}

StackingAction::~StackingAction()
{}

G4ClassificationOfNewTrack StackingAction::ClassifyNewTrack(const G4Track* aTrack)
{
  if (aTrack->GetTrackID() == 1) return fUrgent;

  if (aTrack->GetParentID() == 1 && fDeathTime < 0.) {
    G4Track* parentTrack = G4EventManager::GetEventManager()->GetTrackingManager()->GetTrack();
    fDeathTime = parentTrack->GetGlobalTime();
  }

  if (aTrack->GetGlobalTime() < fDeathTime) return fKill;

  G4double lifeTime = aTrack->GetDefinition()->GetPDGLifeTime();
  if (lifeTime > 1.0 * us || lifeTime < 0.0) {
    G4double kineticEnergy = aTrack->GetKineticEnergy();
    if (kineticEnergy > 1.0 * MeV) {
      G4ThreeVector momentumDirection = aTrack->GetMomentumDirection();
      G4ThreeVector position = aTrack->GetPosition() / m;
      RunAction::AddSecondaryParticle(
        ParticleData {
          aTrack->GetParticleDefinition()->GetParticleName(),
          aTrack->GetParticleDefinition()->GetPDGEncoding(),
          aTrack->GetParticleDefinition()->GetPDGMass(),
          aTrack->GetParticleDefinition()->GetPDGCharge(),
          aTrack->GetKineticEnergy() / MeV,
          std::vector<double> {momentumDirection.x(), momentumDirection.y(), momentumDirection.z()},
          std::vector<double> {position.x(), position.y(), position.z()},
          "Init"
        }
      );
    }
    return fKill;
  }

  return fUrgent;
}

void StackingAction::PrepareNewEvent()
{
  fDeathTime = -1.;
}

}
