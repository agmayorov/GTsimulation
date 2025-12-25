#pragma once

#include <FTFP_BERT.hh>

#include <G4SystemOfUnits.hh>
#include <G4RadioactiveDecayPhysics.hh>

namespace MatterLayer
{

class PhysicsList : public FTFP_BERT
{
  public:
    PhysicsList();
    ~PhysicsList();
};

}
