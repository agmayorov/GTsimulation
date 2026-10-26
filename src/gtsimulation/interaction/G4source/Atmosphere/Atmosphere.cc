#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "G4RunManager.hh"
#include "CLHEP/Random/MixMaxRng.h"

#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"
#include "SimConfig.hh"
#include "RunResult.hh"

#include <streambuf>

namespace py = pybind11;
using namespace Atmosphere;

class NullBuffer : public std::streambuf {
  public:
    int overflow(int c) override { return c; }
};

// Geant4 kernel can be initialized only once per process,
// so the function runs a single event and must not be called again in the same process
RunResult run(
  long seed,
  int pdg, double energy,
  std::array<double, 3> coordinates,
  std::array<double, 3> direction,
  double earth_radius,
  int doy, double sec,
  double lat, double lon,
  double f107a, double f107, double ap
) {
  CLHEP::HepRandom::setTheEngine(new CLHEP::MixMaxRng());
  CLHEP::HepRandom::setTheSeed(seed);

  nrlmsise_input input = {
    0,                        /* year, currently ignored */
    doy,                      /* day of year */
    sec,                      /* seconds in day (UT) */
    0,                        /* altitude in kilometers */
    lat,                      /* geodetic latitude */
    lon,                      /* geodetic longitude */
    sec / 3600. + lon / 15.,  /* local apparent solar time (hours) */
    f107a,                    /* 81 day average of F10.7 flux (centered on doy) */
    f107,                     /* daily F10.7 flux for previous day */
    ap,                       /* magnetic index(daily) */
    nullptr,                  /* ap_array */
  };

  SimConfig simConfig;
  simConfig.particlePDG = pdg;
  simConfig.energy = energy;
  simConfig.coordinates = G4ThreeVector(coordinates[0], coordinates[1], coordinates[2]);
  simConfig.direction = G4ThreeVector(direction[0], direction[1], direction[2]);
  RunResult result;

  // Suppress G4 output during initialization
  NullBuffer nullBuffer;
  std::streambuf* oldCout = std::cout.rdbuf();
  std::cout.rdbuf(&nullBuffer);

  // Construct RunManager and initialize G4 kernel
  G4RunManager* runManager = new G4RunManager();
  runManager->SetUserInitialization(new DetectorConstruction(earth_radius, input));
  runManager->SetUserInitialization(new PhysicsList());
  runManager->SetUserInitialization(new ActionInitialization(simConfig, result));
  runManager->Initialize();

  std::cout.rdbuf(oldCout);

  runManager->BeamOn(1);

  // Job termination
  delete runManager;

  return result;
}

PYBIND11_MODULE(atmosphere, m) {
  m.doc() = "Geant4 Atmosphere Simulator";

  // Exporting data structures
  py::class_<ParticleData>(m, "ParticleData")
    .def_readonly("name", &ParticleData::name)
    .def_readonly("pdgCode", &ParticleData::pdgCode)
    .def_readonly("mass", &ParticleData::mass)
    .def_readonly("charge", &ParticleData::charge)
    .def_readonly("kineticEnergy", &ParticleData::kineticEnergy)
    .def_readonly("momentumDirection", &ParticleData::momentumDirection)
    .def_readonly("position", &ParticleData::position)
    .def_readonly("vertexKineticEnergy", &ParticleData::vertexKineticEnergy)
    .def_readonly("vertexMomentumDirection", &ParticleData::vertexMomentumDirection)
    .def_readonly("vertexPosition", &ParticleData::vertexPosition)
    .def_readonly("lastProcess", &ParticleData::lastProcess);

  py::class_<RunResult>(m, "RunResult")
    .def_readonly("primary", &RunResult::primary)
    .def_readonly("secondaries", &RunResult::secondaries);

  // Exporting the main function
  m.def("run", &run,
        py::arg("seed"), py::arg("pdg"), py::arg("energy"), py::arg("coordinates"), py::arg("direction"),
        py::arg("earth_radius"), py::arg("doy"), py::arg("sec"), py::arg("lat"), py::arg("lon"),
        py::arg("f107a"), py::arg("f107"), py::arg("ap"),
        "Run one event with given parameters (only once per process)");
}
