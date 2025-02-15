import trimesh
mesh = trimesh.load("/home2/zxp/Projects/FoundationPose/demo_data/square_table_leg/mesh/square_table_leg_old.obj")
mesh.apply_scale(100)
mesh.export("/home2/zxp/Projects/FoundationPose/demo_data/square_table_leg/mesh/square_table_leg.obj")