#include "PrimaryGeneratorAction.hh"

namespace Atmosphere
{

PrimaryGeneratorAction::PrimaryGeneratorAction(G4int particlePDG, G4double energy, G4double height, G4double alpha)
: G4VUserPrimaryGeneratorAction(),
  fParticleGun(0),
  fParticlePDG(particlePDG),
  fEnergy(energy),
  fHeight(height),
  fAlpha(alpha)
{
  fParticleGun = new G4ParticleGun();
  
  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(sin(fAlpha),0.,-cos(fAlpha)));
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

  fParticleGun->SetParticleDefinition(particle);
  fParticleGun->SetParticlePosition(G4ThreeVector(0.,0.,fHeight*km));

  fParticleGun->GeneratePrimaryVertex(anEvent);
}

}
