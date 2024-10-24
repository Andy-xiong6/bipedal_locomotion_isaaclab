[INFO] Command Manager:  <CommandManager> contains 1 active terms.
+------------------------------------------------+
|              Active Command Terms              |
+-------+---------------+------------------------+
| Index | Name          |          Type          |
+-------+---------------+------------------------+
|   0   | base_velocity | UniformVelocityCommand |
+-------+---------------+------------------------+

[INFO] Action Manager:  <ActionManager> contains 1 active terms.
+----------------------------------+
|  Active Action Terms (shape: 6)  |
+--------+------------+------------+
| Index  | Name       |  Dimension |
+--------+------------+------------+
|   0    | joint_pos  |          6 |
+--------+------------+------------+
1+3+6+6+6+3=25 +2 height =27
[INFO] Observation Manager: <ObservationManager> contains 1 groups.
+-------------------------------------------------------+
| Active Observation Terms in Group: 'policy' (shape: (24,)) |
+-------------+---------------------------+-------------+
|    Index    | Name                      |    Shape    |
+-------------+---------------------------+-------------+
|      0      | base_lin_vel              |     (3,)    |
|      1      | base_ang_vel              |     (3,)    |
|      2      | proj_gravity              |     (3,)    |
|      3      | vel_command               |     (3,)    |
|      4      | joint_pos                 |     (6,)    |
|      5      | last_action               |     (6,)    |
+-------------+---------------------------+-------------+
[INFO] Event Manager:  <EventManager> contains 3 active terms.
+-------------------------------------+
| Active Event Terms in Mode: 'startup' |
+-----------+-------------------------+
|   Index   | Name                    |
+-----------+-------------------------+
|     0     | add_base_mass           |
+-----------+-------------------------+
+-------------------------------------+
| Active Event Terms in Mode: 'reset' |
+---------+---------------------------+
|  Index  | Name                      |
+---------+---------------------------+
|    0    | reset_robot_base          |
|    1    | reset_robot_joints        |
+---------+---------------------------+
+----------------------------------------------+
|    Active Event Terms in Mode: 'interval'    |
+-------+------------+-------------------------+
| Index | Name       | Interval time range (s) |
+-------+------------+-------------------------+
|   0   | push_robot |       (10.0, 15.0)      |
+-------+------------+-------------------------+

[INFO] Termination Manager:  <TerminationManager> contains 2 active terms.
+---------------------------------+
|     Active Termination Terms    |
+-------+--------------+----------+
| Index | Name         | Time Out |
+-------+--------------+----------+
|   0   | time_out     |   True   |
|   1   | base_contact |  False   |
+-------+--------------+----------+

[INFO] Reward Manager:  <RewardManager> contains 9 active terms.
+---------------------------------------+
|          Active Reward Terms          |
+-------+---------------------+---------+
| Index | Name                |  Weight |
+-------+---------------------+---------+
|   0   | rew_lin_vel_xy      |     2.0 |
|   1   | rew_ang_vel_z       |     1.5 |
|   2   | pen_joint_deviation |   -0.05 |
|   3   | pen_feet_slide      |    -0.1 |
|   4   | pen_lin_vel_z       |    -1.0 |
|   5   | pen_ang_vel_xy      |   -0.05 |
|   6   | pen_action_rate     |   -0.01 |
|   7   | pen_joint_accel     |  -1e-06 |
|   8   | pen_joint_powers    | -0.0005 |
+-------+---------------------+---------+

[INFO] Curriculum Manager:  <CurriculumManager> contains 0 active terms.
+----------------------+
| Active Curriculum Terms |
+-----------+----------+
|   Index   | Name     |
+-----------+----------+
+-----------+----------+

[INFO]: Completed setting up the environment...
Actor MLP: Sequential(
  (0): Linear(in_features=24, out_features=128, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=128, out_features=64, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=64, out_features=32, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=32, out_features=6, bias=True)
)
Critic MLP: Sequential(
  (0): Linear(in_features=24, out_features=128, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=128, out_features=64, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=64, out_features=32, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=32, out_features=1, bias=True)
)
Setting seed: 42
