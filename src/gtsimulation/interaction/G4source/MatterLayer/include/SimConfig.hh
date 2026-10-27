#pragma once

namespace MatterLayer
{

struct SimConfig {
  int particlePDG = 2212;
  double energy = 100.0; // in MeV
  double decayTime = 1e-6; // in seconds
};

}
