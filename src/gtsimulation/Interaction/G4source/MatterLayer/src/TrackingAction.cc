#include "TrackingAction.hh"

namespace MatterLayer
{

TrackingAction::TrackingAction()
{}

TrackingAction::~TrackingAction()
{}

void TrackingAction::PostUserTrackingAction(const G4Track *aTrack)
{
  std::cout.precision(16);

  if (aTrack->GetTrackID() == 1) {
    if (aTrack->GetStep()->GetPostStepPoint()->GetStepStatus() == fWorldBoundary) {
      G4cout << "The primary particle reached boundary of the layer" << G4endl;
    } else {
      G4cout << "The primary particle has died" << G4endl;
    }
    std::cout << "\nInformation about the primary particle:\n"
              << "Name,PDGcode,Mass,Charge,KineticEnergy[MeV],MomentumDirection,Position[m],LastProcess\n"
              << aTrack->GetParticleDefinition()->GetParticleName() << ","
              << aTrack->GetParticleDefinition()->GetPDGEncoding() << ","
              << aTrack->GetParticleDefinition()->GetPDGMass() << ","
              << aTrack->GetParticleDefinition()->GetPDGCharge() << ","
              << aTrack->GetKineticEnergy() / MeV << ","
              << aTrack->GetMomentumDirection() << ","
              << aTrack->GetPosition() / m << ","
              << aTrack->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() << std::endl;
  }
}

}
