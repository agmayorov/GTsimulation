#pragma once

#include <G4VUserActionInitialization.hh>
#include "SimConfig.hh"

#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "StackingAction.hh"
#include "TrackingAction.hh"

namespace MatterLayer
{

class ActionInitialization : public G4VUserActionInitialization
{
  public:
    ActionInitialization(const SimConfig* config);
    ~ActionInitialization();
    void Build() const override;

  private:
    const SimConfig* fConfig;
};

}
