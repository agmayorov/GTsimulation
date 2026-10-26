#include "ActionInitialization.hh"

namespace Atmosphere
{

ActionInitialization::ActionInitialization(const SimConfig& config, RunResult& result)
: fConfig(config),
  fResult(result)
{}

ActionInitialization::~ActionInitialization()
{}

void ActionInitialization::Build() const
{
  SetUserAction(new PrimaryGeneratorAction(fConfig));
  SetUserAction(new TrackingAction(fResult));
}

}
