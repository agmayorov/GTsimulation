#include "PhysicsList.hh"

namespace DecayGenerator
{

PhysicsList::PhysicsList()
{
  // SetVerboseLevel(0);
  RegisterPhysics(new G4DecayPhysics());
  RegisterPhysics(new G4RadioactiveDecayPhysics(0));
  // auto phys = new G4RadioactiveDecayPhysics();
  // RegisterPhysics(phys);
}

PhysicsList::~PhysicsList()
{}

}
