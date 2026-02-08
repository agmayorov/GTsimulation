#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <G4RunManager.hh>

#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"

#include <streambuf>
#include "RunAction.hh"
#include "SimConfig.hh"

namespace py = pybind11;
using namespace MatterLayer;

class NullBuffer : public std::streambuf {
  public:
    int overflow(int c) override { return c; }
};

class Simulator {
  public:
    Simulator(long seed) {
      CLHEP::HepRandom::setTheEngine(new CLHEP::RanecuEngine());
      CLHEP::HepRandom::setTheSeed(seed);

      // Suppress G4 output during initialization
      NullBuffer nullBuffer;
      std::streambuf* oldCout = std::cout.rdbuf();
      std::cout.rdbuf(&nullBuffer);

      // Construct RunManager and initialize G4 kernel
      runManager = new G4RunManager();
      detectorConstruction = new DetectorConstruction();
      runManager->SetUserInitialization(detectorConstruction);
      runManager->SetUserInitialization(new PhysicsList());
      actionInitialization = new ActionInitialization(&simConfig);
      runManager->SetUserInitialization(actionInitialization);
      runManager->Initialize();

      std::cout.rdbuf(oldCout);
    }

    ~Simulator() {
      delete runManager;
    }

    RunResult run(
      int pdg, double energy,
      double mass, double density,
      std::vector<std::string> element_name,
      std::vector<double> element_abundance
    ) {
      double thickness = mass / density / 1e2; // layer thickness in [m]
      detectorConstruction->UpdateParameters(thickness, density, element_name, element_abundance);
      materialCount = detectorConstruction->fMatCounter;
      simConfig.particlePDG = pdg;
      simConfig.energy = energy;
      runManager->BeamOn(1);
      return RunAction::GetResult();
    }

    int materialCount = 0;

  private:
    G4RunManager* runManager;
    DetectorConstruction* detectorConstruction;
    ActionInitialization* actionInitialization;
    SimConfig simConfig;
};

PYBIND11_MODULE(matter_layer, m) {
  m.doc() = "Geant4 Matter Layer Simulator";

  // Exporting data structures
  py::class_<ParticleData>(m, "ParticleData")
    .def_readonly("name", &ParticleData::name)
    .def_readonly("pdgCode", &ParticleData::pdgCode)
    .def_readonly("mass", &ParticleData::mass)
    .def_readonly("charge", &ParticleData::charge)
    .def_readonly("kineticEnergy", &ParticleData::kineticEnergy)
    .def_readonly("momentumDirection", &ParticleData::momentumDirection)
    .def_readonly("position", &ParticleData::position)
    .def_readonly("lastProcess", &ParticleData::lastProcess);

  py::class_<RunResult>(m, "RunResult")
    .def_readonly("primary", &RunResult::primary)
    .def_readonly("secondaries", &RunResult::secondaries);

  // Exporting the main class
  py::class_<Simulator>(m, "Simulator")
    .def(py::init<long>(), py::arg("seed"),
         "Create Simulator with random seed")
    .def("run", &Simulator::run,
         py::arg("pdg"), py::arg("energy"), py::arg("mass"), py::arg("density"),
         py::arg("element_name"), py::arg("element_abundance"),
         "Run one event with given parameters")
    .def_readonly("material_count", &Simulator::materialCount);
}
