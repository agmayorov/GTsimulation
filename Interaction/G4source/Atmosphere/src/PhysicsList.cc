#include "PhysicsList.hh"

namespace Atmosphere
{

PhysicsList::PhysicsList()
: FTFP_BERT(0)
{
  RegisterPhysics(new G4RadioactiveDecayPhysics());
}

PhysicsList::~PhysicsList()
{}

}