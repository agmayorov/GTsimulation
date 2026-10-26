#include "PrimaryGeneratorAction.hh"

#include <G4ParticleGun.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4IonTable.hh>
#include <G4SystemOfUnits.hh>

#include <iostream>

namespace Atmosphere
{

PrimaryGeneratorAction::PrimaryGeneratorAction(const SimConfig& config)
: G4VUserPrimaryGeneratorAction(),
  fConfig(config)
{
  fParticleGun = new G4ParticleGun();
  fParticleGun->SetParticleEnergy(fConfig.energy*MeV);
  fParticleGun->SetParticlePosition(fConfig.coordinates*km);
  fParticleGun->SetParticleMomentumDirection(fConfig.direction);
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
  G4ParticleDefinition* particle = nullptr;
  if (G4ParticleTable::GetParticleTable()->FindParticle(fConfig.particlePDG))
    particle = G4ParticleTable::GetParticleTable()->FindParticle(fConfig.particlePDG);
  else if (G4IonTable::GetIonTable()->GetIon(fConfig.particlePDG))
    particle = G4IonTable::GetIonTable()->GetIon(fConfig.particlePDG);
  else
    std::cerr << "Error: particle was not found in G4ParticleTable and G4IonTable" << std::endl;
  fParticleGun->SetParticleDefinition(particle);
  // Primary vertex
  fParticleGun->GeneratePrimaryVertex(anEvent);
}

}
