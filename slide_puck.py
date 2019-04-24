import gym
import rospy
import numpy as np
from gym import spaces
from ur_gazebo_test2.scripts import ur5_env
# from openai_ros.robot_envs import ur5_env
# from openai_ros.src.openai_ros.robot_envs import ur5_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
# from tf.transformations import euler_from_quaternion
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, GetModelState
from gazebo_msgs.msg import ModelStates

import sys
import universal_robot.ur_kinematics.src.ur_kin_py as ur_kin_py

from puck_sim2.srv import *
from gazebo_msgs.srv import GetWorldProperties

import datetime
global g_max_x
global g_max_y
global g_distance
global goal_x
global goal_y

g_max_x = 0.0
g_max_y = 0.0
timestep_limit_per_episode = 1

register(
    id='UR5Slide-v1',
    entry_point = 'ur_gazebo_test2.scripts.slide_puck:UR5SlidePuckEnv',
    timestep_limit = timestep_limit_per_episode,
)

def best_sol(sols, q_guess, weights):
    valid_sols = []
    for sol in sols:
        test_sol = np.ones(6)*9999.
        for i in range(6):
            for add_ang in [-2.*np.pi, 0, 2.*np.pi]:
                test_ang = sol[i] + add_ang
                if (abs(test_ang) <= 2.*np.pi and
                    abs(test_ang - q_guess[i]) < abs(test_sol[i] - q_guess[i])):
                    test_sol[i] = test_ang
        if np.all(test_sol != 9999.):
            valid_sols.append(test_sol)
    if len(valid_sols) == 0:
        return None
    best_sol_ind = np.argmin(np.sum((weights*(valid_sols - np.array(q_guess)))**2,1))
    return valid_sols[best_sol_ind]


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def rotation(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R


class Param(object):
    pass


def puck_sim_client(obj_init, hand_init, hand_fin):
    rospy.wait_for_service('puck_sim')
    try:
        puck_sim_handle = rospy.ServiceProxy('puck_sim', puck_sim)
        ret = puck_sim_handle(obj_init, hand_init, hand_fin)
        return ret.obj_fin
    except rospy.ServiceException as e:
        print("service failed: %s", e)


class UR5SlidePuckEnv(ur5_env.UR5Env):
    def __init__(self):

        """
        Make UR5 learn how to slide a puck.
        """

        # We execute this one before because there are some functions that this
        # TaskEnv uses that use variables from the parent class, like the effort limit fetch.
        super(UR5SlidePuckEnv, self).__init__()

        # Here we will add any init functions prior to starting the MyRobotEnv

        # Only variable needed to be set here
        global g_max_x
        global g_max_y
        global g_distance
        global goal_x
        global goal_y
        goal_x = 0.43
        goal_y = 1.038
        g_distance = float('inf')

        rospy.logdebug("Start UR5SlidePuckEnv INIT...")
        number_actions = 4

        self.action_space = spaces.Box(low=np.array([0.35, 0.1, np.radians(-3), np.radians(-3)]),
                                       high=np.array([0.4, 0.15, np.radians(3), np.radians(3)]), dtype='float32')

        self._is_action_done = False
        self._has_object = True
        self.param = Param()

        setattr(self.param, 'init_obj_center', Param())
        setattr(self.param, 'obj_center', Param())
        setattr(self.param, 'target_center', Param())
        setattr(self.param, 'target_position', Param())

        setattr(self.param.init_obj_center, 'x', 0.41) #used was 0.45.
        setattr(self.param.init_obj_center, 'y', -0.1)
        setattr(self.param, 'init_obj_width', 0.1)
        setattr(self.param.obj_center, 'x', self.param.init_obj_center.x)
        setattr(self.param.obj_center, 'y', self.param.init_obj_center.y)

        setattr(self.param.target_center, 'x', self.param.init_obj_center.x)
        setattr(self.param.target_center, 'y', self.param.init_obj_center.y + 1.101)#used 0.5
        setattr(self.param, 'target_width', 0.00000002) #used 0.2

        setattr(self.param.target_position, 'x', self.param.target_center.x)
        setattr(self.param.target_position, 'y', self.param.target_center.y)

        temp = SetModelStateRequest()
        self.my_object = temp.model_state.pose
        self.my_target = temp.model_state.pose
        self.my_object.position.x = self.param.init_obj_center.x
        self.my_object.position.y = self.param.init_obj_center.y
        self.my_target.position.x = self.param.target_position.x
        self.my_target.position.y = self.param.target_position.y

        self._initial_T = np.identity(4)
        obs = self._get_obs()   # get observation

        '''
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        '''
        self.init_joints_positions_array = [-60.0* np.pi/180.0, -55.3* np.pi/180.0, 102.5* np.pi/180.0, -47.0* np.pi/180.0, -60.0* np.pi/180.0, -0.6* np.pi/180.0]

        rospy.Subscriber("/gazebo/model_states", ModelStates, self._object_state_callback)

        rospy.wait_for_service('/gazebo/set_model_state')
        #rospy.loginfo("set model_state available")
        self.set_obj_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_world_state = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)

        self.cumulated_steps = 0.0
        self.dynamics_param = True
        self.distance_threshold = 0.10
        # self.reward_type = 'sparse'
        self.reward_type = 'dense'

        self.save_test_log = False
        if self.save_test_log:
            self.log_file = "./log_" + str(datetime.datetime.now()) + ".txt"
            data = "init_x, init_y, target_x, target_y, result_x, result_y\n"
            self.fd = open(self.log_file, 'w')
            self.fd.write(data)
            self.fd.close()

    def my_init(self, dynamics):
        self.dynamics_param = dynamics

    def _set_init_pose(self):

        diff = 100
        while diff >= 1e-1:
            self.joints = []
            self.joints.append(self.init_joints_positions_array)
            self.move_joints_to_angle_blocking(self.joints, 3.0)

            ordered_joint_position = self._get_ordered_joint_attr('position')
            diff = np.linalg.norm(np.array(ordered_joint_position) - np.array(self.init_joints_positions_array))

        return True

    def _set_init_obj_pose(self):
        """
        This function is called 'at _reset_sim() in robot_gazebo_env.py'
        :return:
        """

        # random position
        self.param.obj_center.x = np.random.uniform(self.param.init_obj_center.x - self.param.init_obj_width/2,
                                                    self.param.init_obj_center.x + self.param.init_obj_width/2)
        self.param.obj_center.y = np.random.uniform(self.param.init_obj_center.y - self.param.init_obj_width / 2,
                                                    self.param.init_obj_center.y + self.param.init_obj_width / 2)

        req_position = np.array([self.param.obj_center.x, self.param.obj_center.y, 0.05])
        req_name = 'my_object'
        while not self._set_obj_position(req_name, req_position):
            pass
        #rospy.loginfo("init object position ")

        #rospy.loginfo("position ")

        self.my_object.position.x = self.param.obj_center.x
        self.my_object.position.y = self.param.obj_center.y

        self._init_target()
        #rospy.loginfo("init target done ")
        #print('[Intial position]: ', self.param.obj_center.x, self.param.obj_center.y)
        print('[Intial position]: ', self.my_object.position.x, self.my_object.position.y)


        return True

    def _init_env_variables(self):
        """
        This function is called right after _reset_sim() in robot_gazebo_env.py
        Thus, we know object's initialized position before this function call.
        :return:
        """
        # For Info Purposes
        self.cumulated_stpes = 0.0
        # self._set_init_obj_pose()
        # self._init_target()
        # self._init_target()
        # TODO: set self.target_position to rosparam server for rviz visualization

    def _init_target(self):
        # fixed position
        # self.param.target_position.x = self.param.target_center.x
        # self.param.target_position.y = self.param.target_center.y
        global g_max_x
        global g_max_y
        global g_distance
        global goal_x
        global goal_y

        goal_x = 0.43
        goal_y = 1.038
        g_distance = float('inf')

        # random position
        self.param.target_position.x = np.random.uniform(self.param.target_center.x - self.param.target_width/2,
                                                         self.param.target_center.x + self.param.target_width/2)
        self.param.target_position.y = np.random.uniform(self.param.target_center.y - self.param.target_width/2,
                                                         self.param.target_center.y + self.param.target_width/2)

        req_position = np.array([self.param.target_position.x, self.param.target_position.y, 0.05])
        req_name = 'my_target'
        self._set_obj_position(req_name, req_position)

        self.my_target.position.x = self.param.target_position.x
        self.my_target.position.y = self.param.target_position.y
        # target_position = self.my_target.position
        self.goal = np.array([self.my_target.position.x, self.my_target.position.y])
        print('[Target]: ', self.my_target.position.x, self.my_target.position.y)

        return True

    def _set_obj_position(self, obj_name, position):
        rospy.wait_for_service('/gazebo/set_model_state')
        #rospy.loginfo("set model_state for " + obj_name + " available")
        sms_req = SetModelStateRequest()
        sms_req.model_state.pose.position.x = position[0]
        sms_req.model_state.pose.position.y = position[1]
        sms_req.model_state.pose.position.z = position[2]
        sms_req.model_state.pose.orientation.x = 0.
        sms_req.model_state.pose.orientation.y = 0.
        sms_req.model_state.pose.orientation.z = 0.
        sms_req.model_state.pose.orientation.w = 1.

        sms_req.model_state.twist.linear.x = 0.
        sms_req.model_state.twist.linear.y = 0.
        sms_req.model_state.twist.linear.z = 0.
        sms_req.model_state.twist.angular.x = 0.
        sms_req.model_state.twist.angular.y = 0.
        sms_req.model_state.twist.angular.z = 0.
        sms_req.model_state.model_name = obj_name
        sms_req.model_state.reference_frame = 'world'
        result = self.set_obj_state(sms_req)

        return result.success



    def _set_action(self, action):
        """
        UR5 robot actions
        :type action: object
        :param action:
        :return:
        """

        #action[0] = np.random.uniform(0.35,0.4)
        #action[1] = np.random.uniform(0.1, 0.15)
        #action[2] = np.random.uniform(np.radians(-3), np.radians(3))
        #action[3] = np.random.uniform(np.radians(-3), np.radians(3))


        l = action[0]
        r = action[1]
        theta1 = action[2]
        theta2 = action[3]
        print('[Action] ',l,r,theta1,theta2)
        self._is_action_done = False

        # point target
        p_t = np.array([self.param.target_position.x, self.param.target_position.y])
        # point puck
        p_p = np.array([self.param.obj_center.x, self.param.obj_center.y])
        v = (p_t - p_p)/np.linalg.norm(p_t - p_p, 2)
        R1 = rotation(theta1)
        R2 = rotation(theta2)

        # action end position
        p_e = p_p + R2.dot(r*v)

        # action start position
        p_s = p_p - R1.dot(l*v)
        v2 = (p_e - p_s)/np.linalg.norm(p_e - p_s, 2)

        # test
        # v = np.array([0., 1.0])
        # p_s = np.array([0.65, 0.])
        # v2 = v

        # inverse kin for initial position
        ordered_joint_position = self._get_ordered_joint_attr('position')  # get current joint position
        x = np.identity(4)
        ur_kin_py.forward(ordered_joint_position, x.ravel())
        # set desired position
        x[:2, 3] = p_s
        x[2,3] = -0.02 #used -0.02
        # set desired orientation
        x_hat = np.append(v2, [0.])
        y_hat = np.cross(x[:3,2], x_hat)
        x[:3, 0] = x_hat
        x[:3, 1] = y_hat
        sols = np.zeros([8,6])
        num_sol = ur_kin_py.inverse(x.ravel(), sols.ravel(), float(0.0))
        if num_sol == 0:
            self._is_action_done = True
            return

        qsol = best_sol(sols, ordered_joint_position, [1.]*6)
        self._initial_T = x.copy()
        self._initial_q = qsol.copy()

        # add to joint list
        joint_list = []
        joint_list.append(qsol.tolist())
        # joint_list.append(qsol.tolist())
        # joint_list.append(qsol.tolist())
        self.move_joints_to_angle_blocking(joint_list, 2.0)
        #rospy.loginfo("prepared to action")

        last_joint = joint_list[-1]
        # inverse kin for final position
        xf = x
        N = 2
        joint_list = []
        del_p = p_e - p_s
        for i in range(N):
            p_n = p_s + ((i+1)/N)*del_p
            xf[:2, 3] = p_n
        # xf[2, 3] = -0.1
            sols_f = np.zeros([8,6])
            ur_kin_py.inverse(xf.ravel(), sols_f.ravel(), float(0.0))
            qsol_f = best_sol(sols_f, qsol, [1.]*6)

            # add to joint list
            joint_list.append(qsol_f.tolist())

        # move object to puck_sim result position
        if self.dynamics_param is False:
            ret_pucksim = puck_sim_client(p_p, p_s, p_e)
            result_position = np.append((np.array(ret_pucksim)), 0.05)
            # ret_sop = self._set_obj_position('my_object', result_position)
            while not self._set_obj_position('my_object', result_position):
                pass

        # move
        Tf = 0.5
        timestep = Tf/N
        velocities = []
        temp_vel = []
        final_joint = joint_list[-1]
        for i in range(final_joint.__len__()):
            temp_vel.append((final_joint[i] - last_joint[i])/(0.5*Tf))
        velocities.append(temp_vel)
        velocities.append([0]*6)



        init_time = self.get_world_state().sim_time
        self.move_joints_to_angle_blocking(joint_list, timestep, velocities)
        # self.move_joints_to_angle_blocking(joint_list, timestep)
        self._is_action_done = True

        # wait for seconds
        wait_count = 0
        max_wait_count = 10    # 0.5 * N seconds (default time step is 0.5 sec)
        while self._calc_object_velocity() > 1e-2 and wait_count < max_wait_count:
            joint_list = []
            joint_list.append(qsol_f.tolist())
            self.move_joints_to_angle_blocking(joint_list)
            wait_count += 1


        # self._set_obj_position('my_object', result_position)
        # wait 1 second
        joint_list = []
        joint_list.append(qsol_f.tolist())
        joint_list.append(qsol_f.tolist())
        self.move_joints_to_angle_blocking(joint_list)



    def _is_done(self, observations):
        """
        consider the episode done if:
            1)  the object & the robot stops
            2)  error occurs when robot moves
        :param observations:
        :return:
        """
        if self._is_action_done is True and self._calc_object_velocity() < 1e-2:
            done = True
        else:
            done = False
        return done

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        # initial_pos = np.array([self.my_object.position.x, self.my_object.position.y])
        # d_init = goal_distance(initial_pos, desired_goal)
        print("distance: " + str(d))
        if d < self.distance_threshold:
            return True
        else:
            return False

        #return (d < self.distance_threshold).astype(np.float32)
        # success: 1.0
        # fail: 0.0

    def success_or_failure(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        if d <= self.distance_threshold:
            return True


    def _compute_reward(self, observations, done):
        #d = goal_distance(observations['achieved_goal'], observations['desired_goal'])
        d = goal_distance(observations['achieved_goal'], observations['desired_goal'])
        global g_distance
        #d = g_distance

        # if done:
        #     reward = 1
        # else:
        #     reward = -1

        return -1*d*d

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = goal_distance(achieved_goal, desired_goal)
        global g_distance

        if d < self.distance_threshold:
            return 100
        else:
            return -100

        '''
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
            #return -(d > self.distance_threshold)
        else:
            print('reward', d)
            return -d*d
        '''
    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        :return:
        """
        global g_max_x
        global g_max_y
        global g_distance
        global goal_x
        global goal_y
        T = np.identity(4)      # base to robot end.
        ordered_joint_position = self._get_ordered_joint_attr('position')
        ur_kin_py.forward(ordered_joint_position, T.ravel())
        self.get_end_effector_pose(T)   # get SE3 global fixed frame to robot end. not to end effector
        # obs = np.concatenate([ordered_joint_position, self._end_eff_pose.ravel()])

        # if not hasattr(self, '_initial_T'):
        #     self._initial_T = T
        obs = np.concatenate([self._initial_T.ravel(), self._end_eff_pose.ravel()])  # initial, final end effector pose
        obs = np.append(obs, [self.param.obj_center.x, self.param.obj_center.y,     # initial object position
                              self.my_object.position.x, self.my_object.position.y, g_max_x, g_max_y])    # final object position])
                              #         self.param.obj_center.x, self.param.obj_center.y,     # initial object position
                              # self.my_object.position.x, self.my_object.position.y])    # final object position

        # while not self.my_object:
        #     pass
        compute_re = np.array([g_max_x, g_max_y])

        ach = np.array([self.my_object.position.x, self.my_object.position.y])
        # ach = np.array([0, 0])
        des = np.array([self.param.target_position.x, self.param.target_position.y])
        # des = [0, 0]
        print('[Actual reached]: ', self.my_object.position.x, self.my_object.position.y)

        state = ach
        print('bug for state', state)



        return {
            'observation': obs.copy(),
            'achieved_goal': ach.copy(),
            'desired_goal': des.copy(),
            'compute_reward': compute_re.copy(),
        }



    def _object_state_callback(self, data):
        self.object_states = data
        obj_index = self.object_states.name.index('my_object')
        global g_max_x
        global g_max_y
        global g_distance
        global goal_x
        global goal_y

        d_distance = (self.my_object.position.x - goal_x) * (self.my_object.position.x - goal_x) + (
                self.my_object.position.y - goal_y) * (self.my_object.position.y - goal_y)
        if (d_distance <= g_distance):
            g_distance = d_distance
            g_max_x = self.my_object.position.x
            g_max_y = self.my_object.position.y
        self.my_object = self.object_states.pose[obj_index]
        self.my_object_linear_vel = self.object_states.twist[obj_index].linear
        self.my_object_angular_vel = self.object_states.twist[obj_index].angular
        # rospy.loginfo("x: " + str(self.my_object.position.x))
        # rospy.loginfo("y: " + str(self.my_object.position.y))
        # rospy.loginfo("z: " + str(self.my_object.position.z))

        target_index = self.object_states.name.index('my_target')
        self.my_target = self.object_states.pose[target_index]

    def _calc_object_velocity(self):
        vel = np.linalg.norm([self.my_object_linear_vel.x, self.my_object_linear_vel.y, self.my_object_linear_vel.z])
        return vel



def main():
    rospy.init_node('ur5_slide_puck')

    sys.path.append(" /home/huong/catkin_ws/src/openai_ros/openai_ros/src/openai_ros/task_envs/UR5")
    env = gym.make('UR5Slide-v0')
    rospy.loginfo("GYM environment done")




if __name__ == '__main__':
    main()





