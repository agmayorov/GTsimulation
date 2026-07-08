#include "DetectorConstruction.hh"

namespace DecayGenerator
{

DetectorConstruction::DetectorConstruction()
: G4VUserDetectorConstruction()
{}

DetectorConstruction::~DetectorConstruction()
{}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  //--------------------------------------- World --------------------------------------------------

  G4Orb *world_svol = new G4Orb("world", 1.*mm);

  G4Material *world_mat = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
  G4LogicalVolume *world_lvol = new G4LogicalVolume(world_svol, world_mat, "world");

  G4VPhysicalVolume *world_pvol = new G4PVPlacement(nullptr, G4ThreeVector(), world_lvol, "world", nullptr, false, 0, false);

  return world_pvol;
}

}
