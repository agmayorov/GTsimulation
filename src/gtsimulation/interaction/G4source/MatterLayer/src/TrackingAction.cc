#include "TrackingAction.hh"

namespace MatterLayer
{

TrackingAction::TrackingAction()
{}

TrackingAction::~TrackingAction()
{}

void TrackingAction::PostUserTrackingAction(const G4Track* aTrack)
{
  if (aTrack->GetTrackID() == 1) {
    G4ThreeVector momentumDirection = aTrack->GetMomentumDirection();
    G4ThreeVector position = aTrack->GetPosition() / m;
    RunAction::SetPrimaryParticle(
      ParticleData {
        aTrack->GetParticleDefinition()->GetParticleName(),
        aTrack->GetParticleDefinition()->GetPDGEncoding(),
        aTrack->GetParticleDefinition()->GetPDGMass(),
        aTrack->GetParticleDefinition()->GetPDGCharge(),
        aTrack->GetKineticEnergy() / MeV,
        std::vector<double> {momentumDirection.x(), momentumDirection.y(), momentumDirection.z()},
        std::vector<double> {position.x(), position.y(), position.z()},
        aTrack->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName()
      }
    );
  }
}

}
