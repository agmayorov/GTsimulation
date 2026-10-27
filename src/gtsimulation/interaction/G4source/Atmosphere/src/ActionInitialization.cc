#include "ActionInitialization.hh"

#include "PrimaryGeneratorAction.hh"
#include "TrackingAction.hh"

namespace Atmosphere
{

ActionInitialization::ActionInitialization(const SimConfig& config, RunResult& result)
: fConfig(config),
  fResult(result)
{}

void ActionInitialization::Build() const
{
  SetUserAction(new PrimaryGeneratorAction(fConfig));
  SetUserAction(new TrackingAction(fResult));
}

}
