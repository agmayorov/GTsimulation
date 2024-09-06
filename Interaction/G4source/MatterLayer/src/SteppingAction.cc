#include "SteppingAction.hh"

namespace MatterLayer
{

SteppingAction::SteppingAction()
: fPointOfInteraction(0.,0.,0.),
  fDeltaVector(0.,0.,0.)
{}

SteppingAction::~SteppingAction()
{}

void SteppingAction::UserSteppingAction(const G4Step *step)
{
  std::cout.precision(16);

  if (step->GetTrack()->GetTrackID() == 1 && step->GetTrack()->GetTrackStatus() == 2) {
    if (step->GetPostStepPoint()->GetStepStatus() == 0) {
      std::cout << "The primary particle reached boundary of the layer\n";
    } else {
      std::cout << "The primary particle has interacted\n";
      fPointOfInteraction = step->GetTrack()->GetPosition();
    }
    std::cout << "\nInformation about the primary particle:\n"
              << "Name,PDGcode,Mass,Charge,KineticEnergy[MeV],MomentumDirection,Position[mm],LastProcess\n"
              << step->GetTrack()->GetParticleDefinition()->GetParticleName() << ","
              << step->GetTrack()->GetParticleDefinition()->GetPDGEncoding() << ","
              << step->GetTrack()->GetParticleDefinition()->GetPDGMass() << ","
              << step->GetTrack()->GetParticleDefinition()->GetPDGCharge() << ","
              << step->GetTrack()->GetKineticEnergy() << ","
              << step->GetTrack()->GetMomentumDirection() << ","
              << step->GetTrack()->GetPosition() << ","
              << step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() << std::endl;
    if (step->GetPostStepPoint()->GetStepStatus() != 0 && step->GetNumberOfSecondariesInCurrentStep() > 0)
      std::cout << "\nInformation about the secondary particles:\n"
                << "Name,PDGcode,Mass,Charge,KineticEnergy[MeV],MomentumDirection\n";
  }

  if (step->GetTrack()->GetTrackID() != 1) {
    fDeltaVector = fPointOfInteraction - step->GetPreStepPoint()->GetPosition();
  }

  // Stops the calculation of ionization electrons that occur before the inelastic interaction of the primary particle
  if (step->GetTrack()->GetTrackID() != 1 && fDeltaVector.mag() != 0. && step->GetTrack()->GetParentID() == 1) {
    step->GetTrack()->SetTrackStatus(fKillTrackAndSecondaries);
  }

  if (step->GetTrack()->GetTrackID() != 1 && (fDeltaVector.mag() == 0. || step->GetTrack()->GetParentID() != 1) &&
     (step->GetTrack()->GetParticleDefinition()->GetPDGLifeTime() > 1000. || step->GetTrack()->GetParticleDefinition()->GetPDGLifeTime() == -1.) ) {
    if (step->GetPreStepPoint()->GetKineticEnergy() > 0.5) {
      std::cout << step->GetTrack()->GetParticleDefinition()->GetParticleName() << ","
                << step->GetTrack()->GetParticleDefinition()->GetPDGEncoding() << ","
                << step->GetTrack()->GetParticleDefinition()->GetPDGMass() << ","
                << step->GetTrack()->GetParticleDefinition()->GetPDGCharge() << ","
                << step->GetPreStepPoint()-> GetKineticEnergy() << ","
                << step->GetPreStepPoint()-> GetMomentumDirection() << std::endl;
    }
    step->GetTrack()->SetTrackStatus(fKillTrackAndSecondaries);
  }
}

}
