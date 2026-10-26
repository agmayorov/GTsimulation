#ifndef RunResult_hh
#define RunResult_hh

#include <string>
#include <vector>

namespace Atmosphere
{

struct ParticleData {
  std::string name;
  int pdgCode;
  double mass;
  double charge;
  double kineticEnergy;
  std::vector<double> momentumDirection;       // [x, y, z]
  std::vector<double> position;                // [x, y, z]
  double vertexKineticEnergy;
  std::vector<double> vertexMomentumDirection; // [x, y, z]
  std::vector<double> vertexPosition;          // [x, y, z]
  std::string lastProcess;
};

struct RunResult {
  ParticleData primary;
  std::vector<ParticleData> secondaries;
};

}

#endif
