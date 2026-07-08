#include "TrackingAction.hh"

namespace Atmosphere
{

TrackingAction::TrackingAction()
: fFirstSecondaryParticle(true)
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
  }

  if (aTrack->GetTrackID() != 1 && aTrack->GetKineticEnergy() > 1. &&
      aTrack->GetStep()->GetPostStepPoint()->GetStepStatus() == fWorldBoundary) {
    if (fFirstSecondaryParticle) {
      G4cout << "\nInformation about the secondary particles:\n"
             << "Name,PDGcode,Mass,Charge,Position[m],MomentumDirection,KineticEnergy[MeV],"
             << "VertexPosition[m],VertexMomentumDirection,VertexKineticEnergy[MeV]" << G4endl;
      fFirstSecondaryParticle = false;
    }
    std::cout << aTrack->GetParticleDefinition()->GetParticleName() << ","
              << aTrack->GetParticleDefinition()->GetPDGEncoding() << ","
              << aTrack->GetParticleDefinition()->GetPDGMass() << ","
              << aTrack->GetParticleDefinition()->GetPDGCharge() << ","
              << aTrack->GetPosition()/m << ","
              << aTrack->GetMomentumDirection() << ","
              << aTrack->GetKineticEnergy()/MeV << ","
              << aTrack->GetVertexPosition()/m << ","
              << aTrack->GetVertexMomentumDirection() << ","
              << aTrack->GetVertexKineticEnergy()/MeV << std::endl;
  }
}

}
