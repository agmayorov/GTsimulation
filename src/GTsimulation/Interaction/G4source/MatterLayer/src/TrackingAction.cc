#include "TrackingAction.hh"

namespace MatterLayer
{

TrackingAction::TrackingAction()
: fFirstSecondaryParticle(true),
  fDeathTime(0.)
{}

void TrackingAction::PreUserTrackingAction(const G4Track *aTrack)
{
  // Stops the calculation of ionization electrons that occur before the inelastic interaction of the primary particle
  if (aTrack->GetTrackID() != 1 && (aTrack->GetGlobalTime() < fDeathTime || fDeathTime == 0.))
    const_cast<G4Track*>(aTrack)->SetTrackStatus(fKillTrackAndSecondaries);

  // Print information about long-lived secondary particles after inelastic interaction
  if (aTrack->GetTrackID() != 1 && (aTrack->GetGlobalTime() >= fDeathTime && fDeathTime > 0.) &&
     (aTrack->GetParticleDefinition()->GetPDGLifeTime() == -1. || aTrack->GetParticleDefinition()->GetPDGLifeTime() > 1000.)) {
    if (aTrack->GetKineticEnergy() > 1.*MeV) {
      if (fFirstSecondaryParticle) {
        G4cout << "\nInformation about the secondary particles:\n"
               << "Name,PDGcode,Mass,Charge,KineticEnergy[MeV],MomentumDirection" << G4endl;
        fFirstSecondaryParticle = false;
      }
      std::cout << aTrack->GetParticleDefinition()->GetParticleName() << ","
                << aTrack->GetParticleDefinition()->GetPDGEncoding() << ","
                << aTrack->GetParticleDefinition()->GetPDGMass() << ","
                << aTrack->GetParticleDefinition()->GetPDGCharge() << ","
                << aTrack->GetKineticEnergy()/MeV << ","
                << aTrack->GetMomentumDirection() << std::endl;
    }
    const_cast<G4Track*>(aTrack)->SetTrackStatus(fKillTrackAndSecondaries);
  }
}

void TrackingAction::PostUserTrackingAction(const G4Track *aTrack)
{
  std::cout.precision(16);

  if (aTrack->GetTrackID() == 1) {
    if (aTrack->GetStep()->GetPostStepPoint()->GetStepStatus() == fWorldBoundary) {
      G4cout << "The primary particle reached boundary of the layer" << G4endl;
    } else {
      G4cout << "The primary particle has died" << G4endl;
      fDeathTime = aTrack->GetGlobalTime();
    }
    std::cout << "\nInformation about the primary particle:\n"
              << "Name,PDGcode,Mass,Charge,KineticEnergy[MeV],MomentumDirection,Position[m],LastProcess\n"
              << aTrack->GetParticleDefinition()->GetParticleName() << ","
              << aTrack->GetParticleDefinition()->GetPDGEncoding() << ","
              << aTrack->GetParticleDefinition()->GetPDGMass() << ","
              << aTrack->GetParticleDefinition()->GetPDGCharge() << ","
              << aTrack->GetKineticEnergy()/MeV << ","
              << aTrack->GetMomentumDirection() << ","
              << aTrack->GetPosition()/m << ","
              << aTrack->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() << std::endl;
  }
}

}
