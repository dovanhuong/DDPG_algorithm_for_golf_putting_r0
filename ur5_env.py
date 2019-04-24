import gym
import rospy
import actionlib
import time
import numpy as np
import math

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

from control_msgs.msg import *

from gym import utils, spaces
from gym.utils import seeding


from ur_gazebo_test2.scripts import robot_gazebo_env

from trajectory_msgs.msg import *

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class UR5Env(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self):
        """
        Initializes a new UR5 environment
        """

        rospy.logdebug("Start UR5Env INIT...")

        parameters = rospy.get_param(None)
        index = str(parameters).find('prefix')
        if (index > 0):
            prefix = str(parameters)[index + len("prefix': '"):(
                    index + len("prefix': '") + str(parameters)[index + len("prefix': '"):-1].find("'"))]
            for i, name in enumerate(JOINT_NAMES):
                JOINT_NAMES[i] = prefix + name
        # Variables that we give through the constructor.

        # Internal Vars
        self.controllers_list = ['joint_state_controller',
                                 'arm_controller']

        self.robot_name_space = ""

        reset_controls_bool = True

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv

        super(UR5Env, self).__init__(controllers_list=self.controllers_list,
                                         robot_name_space=self.robot_name_space,
                                         reset_controls=reset_controls_bool)

        # self.gazebo = GazeboConnection()
        rospy.logdebug("UR5Env unpause...")
        self.gazebo.unpauseSim()

        self._check_all_systems_ready()

        # Subscribers
        # to get joint positions
        rospy.Subscriber("/robot/joint_states", JointState, self._joint_state_callback)
        # rospy.Subscriber("/camera/depth/image_row", Image, self._camera_depth_image_raw_callback)
        rospy.Subscriber("/arm_controller/state", JointTrajectoryControllerState, self._joint_traj_cont_state_callback)
        # to get object position in gazebo
        # rospy.Subscriber("/object/joint_states", JointState, self._obj_joint_state_callback)
        # to get camera information in real world
        # rospy.Subscriber("/camera/depth/image_raw", Image, self._camera_dimage_callback)

        # Publishers
        # publish to gazebo environment  # NOT UR SIMULATOR
        self.publishers_array = []
        # self._span_joint_pub = rospy.Publisher('/ur5/span_joint_position_controller/command', Float64, queue_size=1)
        # self._slift_joint_pub = rospy.Publisher('/ur5/slift_joint_position_controller/command', Float64, queue_size=1)
        # self._e_joint_pub = rospy.Publisher('/ur5/e_joint_position_controller/command', Float64, queue_size=1)
        # self._w1_joint_pub = rospy.Publisher('/ur5/w1_joint_position_controller/command', Float64, queue_size=1)
        # self._w2_joint_pub = rospy.Publisher('/ur5/w2_joint_position_controller/command', Float64, queue_size=1)
        # self._w3_joint_pub = rospy.Publisher('/ur5/w3_joint_position_controller/command', Float64, queue_size=1)
        #
        # self.publishers_array.append(self._span_joint_pub)
        # self.publishers_array.append(self._slift_joint_pub)
        # self.publishers_array.append(self._e_joint_pub)
        # self.publishers_array.append(self._w1_joint_pub)
        # self.publishers_array.append(self._w2_joint_pub)
        # self.publishers_array.append(self._w3_joint_pub)

        # self._check_all_publishers_ready()

        rospy.loginfo("all check done. PAUSE")
        self._setup_movement_system()
        rospy.loginfo("movement setup done.")
        rospy.sleep(1)
        # rospy.loginfo((self.get_joint_state()).position[0])
        self.gazebo.pauseSim()

        self._base_pose = np.identity(4)
        self._base_pose[2,3] = 0.1          # defined in universal_robot/ur_gazebo/launch/ur5.launch
        self._end_eff_pose = np.identity(4)
        self._joint_order = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        # self.initial_state = 0
        # obs = self._get_obs()

        self.feedback_count = 0
        rospy.logdebug("Finished UR5Env INIT...")


    def _check_all_systems_ready(self):
        """
        Check that all the sensors, publishers and other simulation systems are operational.
        :return:
        """
        rospy.logdebug("UR5Env check_all_systems_ready...")
        self._check_all_sensors_ready()
        rospy.logdebug("END UR5Env check_all_systems_ready...")
        return True

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_joint_state_ready()
        # TODO: add object's position sensor for gazebo
        # TODO: add camera sensor for real data
        rospy.logdebug("ALL SENSORS READY")

    def _check_joint_state_ready(self):
        self.joint_states = None
        rospy.logdebug("waiting for /joint_states to be READY...")
        while self.joint_states is None and not rospy.is_shutdown():
            try:
                self.joint_states = rospy.wait_for_message("/robot/joint_states", JointState, timeout=5.0)
                rospy.logdebug("Current /joint_states READY=>")

            except rospy.ROSException:
                rospy.logerr("Current /joint_states not ready yet, retrying for getting joint_states")
        return self.joint_states


    def _joint_state_callback(self, data):
        self.joint_states = data

    def _joint_traj_cont_state_callback(self, data):
        self.joint_traj_cont_states = data

    def _setup_tf_listener(self):
        """
        Set ups the TF listener for getting the transforms you ask for.
        """
        # self.listener = tf.TransformListener()

    def _setup_movement_system(self):
        """
        Setup of the movement system.
        :return:
        """
        self.traj_object = UR5ExecTrajectory()


    def _check_all_publishers_ready(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rospy.logdebug("Start all publishers READY")
        for publisher_object in self.publishers_array:
            self._check_pub_connection(publisher_object)
        rospy.logdebug("All publishers READY")

    def _check_pub_connection(self, publisher_object):
        rate = rospy.Rate(10)   # 10 Hz
        while publisher_object.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No subscribers to publisher_object yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("publisher_object Publisher Connected")
        rospy.logdebug("All Publishers READY")

    def _get_ordered_joint_attr(self, attr):
        """
        attr can be
            'position', 'velocity', 'effort'
        :param attr:
        :return:
        """
        data_src = getattr(self.joint_states, attr)
        data = ()
        for i in self._joint_order:
            index = self.joint_states.name.index(i)
            data = data + (data_src[index],)

        return data

    def get_end_effector_pose(self, T):
        """
        calc transform from global fixed frame to robot's end frame
        TODO:
            add transform from robot's end frame to end effector
        :param T:
        :return:
        """
        self._end_eff_pose = np.matmul(self._base_pose, T)

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        # self._check_pub_connection()
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
           define in task_env
        """
        raise NotImplementedError()

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Calculates the reward to give based on the observations given.
                """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------

    def move_joints_to_angle_blocking(self, joints_positions_array, timestep = 0.5, velocities=[]):
        """
        """
        self.traj_object.send_joints_positions(joints_positions_array, timestep, velocities)

    def get_tf_start_to_end_frames(self, start_frame_name, end_frame_name):
        pass

    # def get_camera_depth_image_raw(self):
    #     return self.camera_depth_image_raw

    def get_joint_state(self):
        return self.joint_states


class UR5ExecTrajectory(object):

    def __init__(self):
        # create the connection to the action server
        self.client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        # waits until the action server is up and running
        self.client.wait_for_server()

        self._joint_order = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        self.init_goal_message()

    def _get_ordered_joint_attr(self, joint_states, attr):
        """
        attr can be
            'position', 'velocity', 'effort'
        :param attr:
        :return:
        """
        data_src = getattr(joint_states, attr)
        data = ()
        for i in self._joint_order:
            index = joint_states.name.index(i)
            data = data + (data_src[index],)

        return data

    def init_goal_message(self):

        self.PENDING = 0
        self.ACTIVE = 1
        self.DONE = 2
        self.WARN = 3
        self.ERROR = 4

        # We Initialise the GOAL SYETS GOINT TO INIT POSE
        # creates a goal to send to the action server
        # self.goal = FollowJointTrajectoryGoal()

        # We fill in the Goal

        # self.goal.trajectory = JointTrajectory()

        # self.goal.trajectory.header.stamp = rospy.Time.now()
        # self.goal.trajectory.header.frame_id = "ur5_link_base"
        # self.goal.trajectory.joint_names = ['shoulder_pan_joint',
        #                                     'shoulder_lift_joint',
        #                                     'elbow_joint',
        #                                     'wrist_1_joint',
        #                                     'wrist_2_joint',
        #                                     'wrist_3_joint']
        # self.goal.trajectory.joint_names = JOINT_NAMES

        self.max_values = [ math.pi,
                            math.pi,
                            math.pi,
                            math.pi,
                            math.pi,
                            math.pi]

        self.min_values = [-math.pi,
                           -math.pi,
                           -math.pi,
                           -math.pi,
                           -math.pi,
                           -math.pi]

        # self.goal.trajectory.points = []
        # joint_traj_point = JointTrajectoryPoint()
        #
        # # TODO
        # joint_traj_point.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # # joint_traj_point.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # # joint_traj_point.accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # joint_traj_point.time_from_start = rospy.Duration(1.0)
        #
        # self.goal.trajectory.points.append(joint_traj_point)
        # self.goal.trajectory.points = [JointTrajectoryPoint(positions=joint_traj_point.positions,
        #                                                     velocities=[0]*6,
        #                                                     time_from_start=rospy.Duration(1.0))]

    def get_goal(self):
        return self.goal

    def feedback_callback(self, feedback):

        if (self.feedback_count % 10 == 0):
            #rospy.loginfo("#### FEEDBACK ####")
            rospy.loginfo(str(feedback.error.positions))
            #rospy.loginfo("#### ###### ######")
        self.feedback_count = self.feedback_count + 1

    def send_joints_positions(self, joints_positions_array, seconds_duration=0.3, joints_velocity_array=[]):
        """
        :type seconds_duration: object
        """
        # # create the connection to the action server
        # self.client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory',
        #                                            FollowJointTrajectoryAction)
        # self.client.cancel_all_goals()
        # # waits until the action server is up and running
        # ret = self.client.wait_for_server()

        parameters = rospy.get_param(None)
        self.goal = FollowJointTrajectoryGoal()
        self.goal.trajectory = JointTrajectory()
        self.goal.trajectory.joint_names = JOINT_NAMES
        # my_goal = self.get_goal()
        # my_goal = FollowJointTrajectoryGoal()
        # my_goal.trajectory = JointTrajectory()
        # my_goal.trajectory.joint_names = JOINT_NAMES

        joint_states = rospy.wait_for_message("/robot/joint_states", JointState)
        # joint_pos = joint_states.position

        positions = self._get_ordered_joint_attr(joint_states, 'position')
        # positions = [joint_pos[joint_states.name.index(joint)] for joint in self.goal.trajectory.joint_names]
        # curr_time = rospy.Time.now() + rospy.Duration(10.0)
        curr_time = rospy.Duration(0.0)
        self.goal.trajectory.points = [JointTrajectoryPoint(positions=positions,
                                                            velocities=[]*6,
                                                            time_from_start=rospy.Duration(0.0))]
        self.goal.trajectory.points = []

        # clamp joint limits
        for i in range(joints_positions_array.__len__()):
            # joint_traj_point = JointTrajectoryPoint()
            # joint_traj_point.positions = np.clip(joints_positions_array[i],
            #                                      self.min_values,
            #                                      self.max_values).tolist()
            # joint_traj_point.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # # joint_traj_point.accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # joint_traj_point.effort = []
            # joint_traj_point.time_from_start = rospy.Duration(seconds_duration*(i+1))
            positions = np.clip(joints_positions_array[i], self.min_values, self.max_values).tolist()
            if not joints_velocity_array:
                velocities = []*6
            else:
                velocities = joints_velocity_array[i]
            d = float(seconds_duration*(i+1))
            self.goal.trajectory.points.append(
                JointTrajectoryPoint(positions=positions, velocities=velocities,
                                     time_from_start=rospy.Duration(d))
            )

            # my_goal.trajectory.points = []
            # my_goal.trajectory.points.append(joint_traj_point)

        # sends the goal to the action server, specifying which feedback function
        # to call when feedback received

        # self.goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(1.0)
        # self.client.send_goal(self.goal, feedback_cb=self.feedback_callback)
        self.feedback_count = 0
        self.client.send_goal(self.goal, feedback_cb=self.feedback_callback)
        # rospy.logwarn(self.client.wait_for_result())

        # Uncomment these lines to test goal preemption:
        # self.client.cancel_goal()  # would cancel the goal 3 seconds after starting


        state_result = self.client.get_state()

        rate = rospy.Rate(10)


        #rospy.loginfo("state_result: "+str(state_result))

        while_count = 0
        while state_result < self.DONE:     # if PENDING or ACTIVE
            #if (while_count%10 == 0):
                #rospy.loginfo("Doing stuff while waiting for the Server to give a result...")
            rate.sleep()
            state_result = self.client.get_state()
            #if (while_count % 10 == 0):
                #rospy.loginfo("state_result: "+str(state_result))
            while_count = while_count + 1

        #rospy.loginfo("[Result] State: "+str(state_result))
        #if state_result == self.ERROR:
            #rospy.logerr("Something went wrong in the Server Side")
        #if state_result == self.WARN:
            #rospy.logwarn("There is a warning in the Server Side")
            




def main():
    rospy.init_node('ur5 environment')

    env = UR5Env()

    return


if __name__ == '__main__':
    main()