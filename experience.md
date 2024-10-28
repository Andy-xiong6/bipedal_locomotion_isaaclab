TASK      EXPERIENCE             Effect                  parameter

flat      2024-10-28_13-30-26    walk good                 --max_iteration=3001 --num_envs=4096 
                                                               
+----------------------------------------------+
|             Active Reward Terms              |
+-------+---------------------------+----------+
| Index | Name                      |   Weight |
+-------+---------------------------+----------+
|   0   | rew_lin_vel_xy            |     10.0 |
|   1   | rew_ang_vel_z             |        5 |
|   2   | rew_no_fly                |      5.0 |
|   3   | pen_feet_air_time         |      5.0 |
|   4   | pen_undesired_contacts    |    -50.0 |
|   5   | pen_lin_vel_z             |     -0.5 |
|   6   | pen_ang_vel_xy            |    -0.05 |
|   7   | pen_action_rate           |    -0.01 |
|   8   | pen_joint_accel           | -2.5e-07 |
|   9   | pen_joint_powers          |  -0.0005 |
|   10  | pen_flat_orientation      |     -5.0 |
|   11  | pen_base_height           |    -10.0 |
|   12  | pen_feet_contact_forces   |    -0.01 |
|   13  | pen_applied_torque_limits |     -0.1 |
|   14  | pen_no_contact            |     -5.0 |
+-------+---------------------------+----------+

rough     2024-10-28_16-15-13    walk good                 --max_iteration=4001 --num_envs=4096

+----------------------------------------------+
|             Active Reward Terms              |
+-------+---------------------------+----------+
| Index | Name                      |   Weight |
+-------+---------------------------+----------+
|   0   | rew_lin_vel_xy            |     10.0 |
|   1   | rew_ang_vel_z             |        5 |
|   2   | rew_no_fly                |      5.0 |
|   3   | pen_feet_air_time         |      5.0 |
|   4   | pen_undesired_contacts    |    -50.0 |
|   5   | pen_lin_vel_z             |     -0.5 |
|   6   | pen_ang_vel_xy            |    -0.05 |
|   7   | pen_action_rate           |    -0.01 |
|   8   | pen_joint_accel           | -2.5e-07 |
|   9   | pen_joint_powers          |  -0.0005 |
|   10  | pen_flat_orientation      |     -5.0 |
|   11  | pen_base_height           |    -10.0 |
|   12  | pen_feet_contact_forces   |    -0.01 |
|   13  | pen_applied_torque_limits |     -0.1 |
|   14  | pen_no_contact            |     -5.0 |
+-------+---------------------------+----------+

