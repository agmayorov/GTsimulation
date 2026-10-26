#pragma once

#include <G4ThreeVector.hh>

namespace Atmosphere
{

struct SimConfig
{
  int particlePDG = 2212;
  double energy = 100.0; // in MeV
  G4ThreeVector coordinates; // in km
  G4ThreeVector direction = G4ThreeVector(0., 0., -1.);
};

}
