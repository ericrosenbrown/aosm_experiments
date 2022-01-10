import time
import thortils as tt
from thortils import constants
from thortils.controller import launch_controller, thor_controller_param
from thortils.map3d import Map3D, Mapper3D
from thortils.utils.visual import GridMapVisualizer
from thortils.agent import thor_reachable_positions
import matplotlib.pyplot as plt

import open3d as o3d

def test_mapper(scene, floor_cut=0.1):
    controller = launch_controller({**constants.CONFIG, **{'scene': scene}})
    mapper = Mapper3D(controller)

    mapper.automate(num_stops=20, sep=1.5)

    grid_map = mapper.get_grid_map(floor_cut=floor_cut, debug=False)

    #o3d.visualization.draw_geometries([mapper.map.pcd])
    mapper.map.to_grid_map(debug=True)

    controller.stop()



if __name__ == "__main__":
    test_mapper("FloorPlan1")
    #test_mapper("FloorPlan201", floor_cut=0.3)

