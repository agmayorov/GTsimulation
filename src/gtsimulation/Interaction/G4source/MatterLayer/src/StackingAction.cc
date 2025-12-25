#include "StackingAction.hh"

namespace MatterLayer
{

StackingAction::StackingAction()
: fFirstSecondaryParticle(true),
  fDeathTime(-1.)
{}

StackingAction::~StackingAction()
{}

G4ClassificationOfNewTrack StackingAction::ClassifyNewTrack(const G4Track *aTrack)
{
  if (aTrack->GetTrackID() == 1) return fUrgent;

  if (aTrack->GetParentID() == 1 && fDeathTime < 0.) {
    G4Track *parentTrack = G4EventManager::GetEventManager()->GetTrackingManager()->GetTrack();
    fDeathTime = parentTrack->GetGlobalTime();
  }

  if (aTrack->GetGlobalTime() < fDeathTime) return fKill;

  G4double lifeTime = aTrack->GetDefinition()->GetPDGLifeTime();
  if (lifeTime > 1.0 * us || lifeTime < 0.0) {
    G4double kineticEnergy = aTrack->GetKineticEnergy();
    if (kineticEnergy > 1.0 * MeV) {
      if (fFirstSecondaryParticle) {
        G4cout << "\nInformation about the secondary particles:\n"
               << "Name,PDGcode,Mass,Charge,KineticEnergy[MeV],MomentumDirection" << G4endl;
        fFirstSecondaryParticle = false;
      }
      std::cout << aTrack->GetParticleDefinition()->GetParticleName() << ","
                << aTrack->GetParticleDefinition()->GetPDGEncoding() << ","
                << aTrack->GetParticleDefinition()->GetPDGMass() << ","
                << aTrack->GetParticleDefinition()->GetPDGCharge() << ","
                << aTrack->GetKineticEnergy() / MeV << ","
                << aTrack->GetMomentumDirection() << std::endl;
    }
    return fKill;
  }

  return fUrgent;
}

}
