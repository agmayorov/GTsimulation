#pragma once

#include <G4VUserPrimaryGeneratorAction.hh>
#include <G4ParticleGun.hh>
#include "SimConfig.hh"

#include <G4SystemOfUnits.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4IonTable.hh>

namespace MatterLayer
{

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction(const SimConfig* config);
    ~PrimaryGeneratorAction();
    void GeneratePrimaries(G4Event* anEvent) override;

  private:
    G4ParticleGun* fParticleGun;
    const SimConfig* fConfig;
};

}
