#include "TrackingAction.hh"

namespace Atmosphere
{

TrackingAction::TrackingAction()
{}

void TrackingAction::PreUserTrackingAction(const G4Track *aTrack)
{}

void TrackingAction::PostUserTrackingAction(const G4Track *aTrack)
{
  std::cout.precision(16);

  if (aTrack->GetTrackID() == 1) {
    std::cout << "\nInformation about the primary particle:\n"
              << "Name,PDGcode,Mass,Charge,PositionInteraction[m],LastProcess\n"
              << aTrack->GetParticleDefinition()->GetParticleName() << ","
              << aTrack->GetParticleDefinition()->GetPDGEncoding() << ","
              << aTrack->GetParticleDefinition()->GetPDGMass() << ","
              << aTrack->GetParticleDefinition()->GetPDGCharge() << ","
              << aTrack->GetPosition()/m << ","
              << aTrack->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() << std::endl;
    G4cout << "\nInformation about the secondary particles:\n"
           << "Name,PDGcode,Mass,Charge,KineticEnergy[MeV],MomentumDirection,Position[m]" << G4endl;
  }

  if (aTrack->GetTrackID() != 1 && aTrack->GetKineticEnergy() > 1. &&
      aTrack->GetStep()->GetPostStepPoint()->GetStepStatus() == fWorldBoundary &&
      aTrack->GetMomentumDirection().z() > 0.) {
    std::cout << aTrack->GetParticleDefinition()->GetParticleName() << ","
              << aTrack->GetParticleDefinition()->GetPDGEncoding() << ","
              << aTrack->GetParticleDefinition()->GetPDGMass() << ","
              << aTrack->GetParticleDefinition()->GetPDGCharge() << ","
              << aTrack->GetKineticEnergy()/MeV << ","
              << aTrack->GetMomentumDirection() << ","
              << aTrack->GetPosition()/m << std::endl;
  }
}

}
