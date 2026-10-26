#pragma once

#include <FTFP_BERT.hh>

namespace MatterLayer
{

class PhysicsList : public FTFP_BERT
{
  public:
    PhysicsList();
    ~PhysicsList() override = default;
    void ConstructProcess() override;
};

}
