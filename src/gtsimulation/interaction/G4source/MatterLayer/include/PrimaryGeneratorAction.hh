#pragma once

#include <G4VUserPrimaryGeneratorAction.hh>
#include "SimConfig.hh"

class G4ParticleGun;

namespace MatterLayer
{

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction(const SimConfig& config);
    ~PrimaryGeneratorAction() override;
    void GeneratePrimaries(G4Event* anEvent) override;

  private:
    G4ParticleGun* fParticleGun;
    const SimConfig& fConfig;
};

}
