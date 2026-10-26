#pragma once

#include <G4VUserActionInitialization.hh>
#include "SimConfig.hh"

namespace MatterLayer
{

class ActionInitialization : public G4VUserActionInitialization
{
  public:
    ActionInitialization(const SimConfig& config);
    ~ActionInitialization() override = default;
    void Build() const override;

  private:
    const SimConfig& fConfig;
};

}
