#include "PrimaryGeneratorAction.hh"

namespace MatterLayer
{

PrimaryGeneratorAction::PrimaryGeneratorAction(G4int particlePDG, G4double energy)
: G4VUserPrimaryGeneratorAction(),
  fParticleGun(0),
  fParticlePDG(particlePDG),
  fEnergy(energy)
{
  fParticleGun = new G4ParticleGun();

  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0.,0.,1.));
  fParticleGun->SetParticleEnergy(fEnergy*MeV);
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event *anEvent)
{
  G4ParticleDefinition *particle = 0;
  if (G4ParticleTable::GetParticleTable()->FindParticle(fParticlePDG))
    particle = G4ParticleTable::GetParticleTable()->FindParticle(fParticlePDG);
  else if (G4IonTable::GetIonTable()->GetIon(fParticlePDG))
    particle = G4IonTable::GetIonTable()->GetIon(fParticlePDG);
  else
    std::cerr << "Error: particle was not found in G4ParticleTable and G4IonTable" << std::endl;

  // Remove "Decay" procecc for
  if (fParticlePDG == 1000010030) // triton
    particle->GetProcessManager()->RemoveProcess(3);
  else if (fParticlePDG == 13 || fParticlePDG == -13) // mu- and mu+
    particle->GetProcessManager()->RemoveProcess(7);
  else if (fParticlePDG == -2112) // antineutron
    particle->GetProcessManager()->RemoveProcess(1);

  fParticleGun->SetParticleDefinition(particle);
  fParticleGun->SetParticlePosition(G4ThreeVector(0.,0.,0.));

  fParticleGun->GeneratePrimaryVertex(anEvent);
}

}
