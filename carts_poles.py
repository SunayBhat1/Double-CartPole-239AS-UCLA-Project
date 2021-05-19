import math
from numpy.lib import angle
import pymunk
import pymunk.constraints
import gym
from gym import spaces, logger
import numpy as np

class CartsPolesEnv(gym.Env):

    def __init__(self, angle=0, dt=1/100, force_mag=5):
        # dt is the simulation step

        #gym.Env.__init__(self)
        super(CartsPolesEnv, self).__init__()
        self._init_objects(angle)
        self.dt = dt
        self.force_mag = force_mag

        self.action_space = spaces.Discrete(9)
        self.action_tuple = ((0,0),(0,1),(1,0),(1,1),(0,-1),(-1,0),(-1,-1),(-1,1),(1,-1))

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 4

        high = np.array([self.x_threshold * 2,
                    np.finfo(np.float32).max,
                    self.x_threshold * 2,
                    np.finfo(np.float32).max,
                    2 * math.pi,
                    np.finfo(np.float32).max,
                    2 * math.pi,
                    np.finfo(np.float32).max,
                    2 * math.pi,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max],
                dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None


    def _init_objects(self,angle):
        self.space = pymunk.Space()
        self.space.gravity = 0, -9.8 #set the gravity of the system
        fil = pymunk.ShapeFilter(group=1) #used to make sure the ground does not collide with others

        # create ground
        self.ground = pymunk.Segment(self.space.static_body, (-4, -0.1), (4, -0.1), 0.1)
        self.ground.friction = 0.1
        self.ground.filter = fil
        self.space.add(self.ground)

        #create cart1
        self.cart1_mass = 0.5*2
        self.cart1_size = 0.3, 0.2
        self.cart1_moment = pymunk.moment_for_box(self.cart1_mass, self.cart1_size)
        self.cart1_body = pymunk.Body(mass=self.cart1_mass, moment=self.cart1_moment)
        self.cart1_body.position = -.75, self.cart1_size[1] / 2
        self.cart1_shape = pymunk.Poly.create_box(self.cart1_body, self.cart1_size)
        self.cart1_shape.friction = self.ground.friction
        self.space.add(self.cart1_body, self.cart1_shape)

        # create cart2
        self.cart2_mass = 0.5*2
        self.cart2_size = 0.3, 0.2
        self.cart2_moment = pymunk.moment_for_box(self.cart2_mass, self.cart2_size)
        self.cart2_body = pymunk.Body(mass=self.cart1_mass, moment=self.cart2_moment)
        self.cart2_body.position = .75, self.cart1_size[1] / 2
        self.cart2_shape = pymunk.Poly.create_box(self.cart2_body, self.cart2_size)
        self.cart2_shape.friction = self.ground.friction
        self.space.add(self.cart2_body, self.cart2_shape)


        # create pendulum (the left one)
        self.pend1_length = 0.6  # to center of mass
        self.pend1_size = 0.01, self.pend1_length * 2  # to get CoM at 0.6 m
        self.pend1_mass = 0.2
        self.pend1_moment = pymunk.moment_for_box(self.pend1_mass, self.pend1_size)
        self.pend1_body = pymunk.Body(mass=self.pend1_mass, moment=self.pend1_moment)
        self.pend1_body.angle = -math.pi/4
        self.pend1_body.position = self.cart1_body.position[0] + self.pend1_length * math.cos(self.pend1_body.angle), \
                                    self.cart1_body.position[1] + self.cart1_size[1] / 2 - self.pend1_length * math.sin(self.pend1_body.angle)
        self.pend1_shape = pymunk.Poly.create_box(self.pend1_body, self.pend1_size)
        self.pend1_shape.filter = fil
        self.space.add(self.pend1_body, self.pend1_shape)
        
        # create pendulum 2 (right)
        self.pend2_length = 0.6  # to center of mass
        self.pend2_size = 0.01, self.pend2_length * 2  # to get CoM at 0.6 m
        self.pend2_mass = .2
        self.pend2_moment = pymunk.moment_for_box(self.pend2_mass, self.pend2_size)
        self.pend2_body = pymunk.Body(mass = self.pend2_mass, moment = self.pend2_moment)
        self.pend2_body.angle = -1*self.pend1_body.angle-math.pi
        self.pend2_body.position = self.cart2_body.position[0] + self.pend2_length * math.cos(self.pend2_body.angle), \
                                    self.cart2_body.position[1] + self.cart2_size[1] / 2 - self.pend2_length * math.sin(self.pend2_body.angle)
        self.pend2_shape = pymunk.Poly.create_box(self.pend2_body, self.pend2_size)
        self.pend2_shape.filter = fil
        self.space.add(self.pend2_body, self.pend2_shape)
        
        # create pendulum 3 (top)
        self.pend3_length = 0.3  # to center of mass
        self.pend3_size = 0.01, self.pend3_length * 2  # to get CoM at 0.6 m
        self.pend3_mass = .4
        self.pend3_moment = pymunk.moment_for_box(self.pend3_mass, self.pend3_size)
        self.pend3_body = pymunk.Body(mass = self.pend3_mass, moment = self.pend3_moment)
        self.pend3_body.angle = angle
        self.pend3_body.position = 0-(self.pend3_length-.05) * math.sin(self.pend3_body.angle), -2 * self.pend1_length * math.sin(self.pend1_body.angle) + self.cart1_size[1] / 2 + (self.pend3_length -.05)* math.cos(self.pend3_body.angle)
        self.pend3_shape = pymunk.Poly.create_box(self.pend3_body, self.pend3_size)
        self.pend3_shape.filter = fil
        self.space.add(self.pend3_body, self.pend3_shape)
        
        # create joint
        self.joint1 = pymunk.constraints.PivotJoint(self.cart1_body, self.pend1_body, self.cart1_body.position + (0, self.cart1_size[1] / 2))
        self.joint1.collide_bodies = False
        self.space.add(self.joint1)
        
        
        # create joint 2
        self.joint2 = pymunk.constraints.PivotJoint(self.cart2_body, self.pend2_body, self.cart2_body.position + (0, self.cart2_size[1] / 2))
        self.joint2.collide_bodies = False
        self.space.add(self.joint2)
        
        # create joint 3
        self.joint3 = pymunk.constraints.PivotJoint(self.pend1_body, self.pend2_body, (0,-2*self.pend1_length*math.sin(self.pend1_body.angle)+self.cart1_size[1]/2))
        self.joint3.collide_bodies = True
        self.space.add(self.joint3)
        
        # create joint 4
        self.joint4 = pymunk.constraints.PivotJoint(self.pend1_body, self.pend3_body, (0,-2*self.pend1_length*math.sin(self.pend1_body.angle)+self.cart1_size[1]/2))
        self.joint4.collide_bodies = True
        self.space.add(self.joint4)
        
        # print(f"cart mass = {self.cart1_body.mass:0.1f} kg")
        # print(f"pendulum mass = {self.pend1_body.mass:0.1f} kg, pendulum moment = {self.pend1_body.moment:0.3f} kg*m^2")
        

    def step(self, action_select):
        """
        Take in and apply actions, step pymunk space, output new state variables, reward, and done
        """
        action = self.action_tuple[action_select]
        force_on_cart1, force_on_cart2 = action[0]*self.force_mag, action[1]*self.force_mag

        self.cart1_body.apply_force_at_local_point((force_on_cart1, 0.0), (0.0, 0.0))
        self.cart2_body.apply_force_at_local_point((force_on_cart2, 0.0), (0.0, 0.0))
        
        self.space.step(self.dt)
        x1 = self.cart1_body.position[0]
        x1_dot = self.cart1_body.velocity[0]
        x2 = self.cart2_body.position[0]
        x2_dot = self.cart2_body.position[0]
        tp = self.pend3_body.angle
        wp = self.pend3_body.angular_velocity

        t1 = self.pend1_body.angle
        w1 = self.pend1_body.angular_velocity

        t2 = self.pend2_body.angle
        w2 = self.pend2_body.angular_velocity

        xp, yp = self.pend3_body.position[0], self.pend3_body.position[1]

        self.state = (x1, x1_dot, x2, x2_dot, t1, w1, t2, w2, tp, wp, xp, yp)
        # self.state = (x1, x1_dot, x2, x2_dot, t1, w1, t2, w2, tp, wp)
        # self.state = (x1, x2, tp)
        
        # print('Pend 3 angle: ' ,self.state[8]*180/math.pi)
        # print('Cart 1 Body ' ,self.state[8]*180/math.pi)
        
        # Stopping condition (angle of pole 3 is in 30:330, ie. over 60 degrees from upright)
        done = bool(
                abs(tp) > math.pi/8
                or yp < self.pend3_length/2
                or x1>=4 or x1<=-4 or x2>=4 or x2<=-4
        )

        # print(done)

        
        # action is the force to two carts, [f1, f2]
        # f1 can be either [1, -1 , 0]

        # abs(np.cos(tp))
        if not done:
            reward = self.dt #abs(np.cos(tp))*self.dt
        else:
            reward = 0

        return np.array(self.state), reward, done, {}

    def reset(self,angle=0):
         if self.space:
             del self.space
         self._init_objects(angle)

         x1 = self.cart1_body.position[0]
         x1_dot = self.cart1_body.velocity[0]
         x2 = self.cart2_body.position[0]
         x2_dot = self.cart2_body.position[0]
         tp = self.pend3_body.angle
         wp = self.pend3_body.angular_velocity

         t1 = self.pend1_body.angle
         w1 = self.pend1_body.angular_velocity

         t2 = self.pend2_body.angle
         w2 = self.pend2_body.angular_velocity

         xp, yp = self.pend3_body.position[0], self.pend3_body.position[1]

         self.state = (x1, x1_dot, x2, x2_dot, t1, w1, t2, w2, tp, wp, xp, yp)

         return np.array(self.state)

    def render(self, mode="human"):
        
        screen_width = 1000
        screen_height = 700

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        PPM = scale

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -self.ground.a.x*PPM, self.ground.a.x *PPM, self.ground.a.y *PPM/ 2, -self.ground.a.y*PPM / 2
            ground= rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.groundtrans = rendering.Transform()
            ground.add_attr(self.groundtrans)
            self.viewer.add_geom(ground)
            l, r, t, b = -self.cart1_size[0]*PPM / 2, self.cart1_size[0] *PPM/ 2, self.cart1_size[1] *PPM/ 2, -self.cart1_size[1]*PPM / 2
            axleoffset = self.cart1_size[1] / 4.0
            cart1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans1 = rendering.Transform()
            cart1.add_attr(self.carttrans1)
            self.viewer.add_geom(cart1)
            l, r, t, b = -self.cart2_size[0]*PPM / 2, self.cart2_size[0] *PPM/ 2, self.cart2_size[1] *PPM/ 2, -self.cart2_size[1]*PPM / 2
            axleoffset = self.cart2_size[1] / 4.0
            cart2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans2 = rendering.Transform()
            cart2.add_attr(self.carttrans2)
            self.viewer.add_geom(cart2)
            l, r, t, b = -self.pend1_size[0]*PPM / 2, self.pend1_size[0] *PPM / 2, self.pend1_size[1] *PPM/ 2, -self.pend1_size[1]*PPM / 2
            pend1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.pendtrans1 = rendering.Transform()
            pend1.add_attr(self.pendtrans1)
            self.viewer.add_geom(pend1)
            l, r, t, b = -self.pend2_size[0]*PPM / 2, self.pend2_size[0] *PPM / 2, self.pend2_size[1] *PPM/ 2, -self.pend2_size[1]*PPM / 2
            pend2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.pendtrans2 = rendering.Transform()
            pend2.add_attr(self.pendtrans2)
            self.viewer.add_geom(pend2)
            l, r, t, b = -self.pend3_size[0]*PPM / 2, self.pend3_size[0] *PPM / 2, self.pend3_size[1] *PPM/ 2, -self.pend3_size[1]*PPM / 2
            pend3 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.pendtrans3 = rendering.Transform()
            pend3.add_attr(self.pendtrans3)
            self.viewer.add_geom(pend3)

        if self.state is None:
            return None
        
        self.groundtrans.set_translation(screen_width / 2.0,screen_height / 2.0)
        cur_state= self.state
        cart1y=self.cart1_body.position[1]*scale + screen_height / 2.0
        cart2y=self.cart2_body.position[1]*scale + screen_height / 2.0
        cart1x = self.cart1_body.position[0]* scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans1.set_translation(cart1x, cart1y)
        cart2x = self.cart2_body.position[0]* scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans2.set_translation(cart2x, cart2y)

        pend1_angle=cur_state[4]
        self.pendtrans1.set_rotation(pend1_angle)
        pend1x=(self.pend1_body.position[0])*scale + screen_width / 2.0
        pend1y=(self.pend1_body.position[1])*scale + screen_height / 2.0
        self.pendtrans1.set_translation(pend1x, pend1y)
        
        pend2_angle=cur_state[6]
        self.pendtrans2.set_rotation(pend2_angle)
        pend2x=(self.pend2_body.position[0])*scale + screen_width / 2.0
        pend2y=(self.pend2_body.position[1])*scale + screen_height / 2.0
        self.pendtrans2.set_translation(pend2x, pend2y)

        pend3_angle=cur_state[8]
        self.pendtrans3.set_rotation(pend3_angle)
        pend3x=(cur_state[10])*scale + screen_width / 2.0
        pend3y=(cur_state[11])*scale + screen_height / 2.0
        self.pendtrans3.set_translation(pend3x, pend3y)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None