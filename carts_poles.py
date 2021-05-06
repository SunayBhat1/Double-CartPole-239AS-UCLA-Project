import math
import pymunk
import pymunk.constraints
import gym

class CartsPolesEnv(gym.Env):

    def __init__(self, dt=1/60, force_mag=10):
        # dt is the simulation step

        #gym.Env.__init__(self)
        super(CartsPolesEnv, self).__init__()
        self._init_objects()
        self.dt = dt
        self.force_mag = 10
        self.action_space = ((0,0),(0,1),(1,0),(1,1))

    def _init_objects(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, -9.8 #set the gravity of the system
        fil = pymunk.ShapeFilter(group=1) #used to make sure the ground does not collide with others

        # create ground
        self.ground = pymunk.Segment(self.space.static_body, (-4, -0.1), (4, -0.1), 0.1)
        self.ground.friction = 0.1
        self.ground.filter = fil
        self.space.add(self.ground)

        #create cart1
        self.cart1_mass = 0.5
        self.cart1_size = 0.3, 0.2
        self.cart1_moment = pymunk.moment_for_box(self.cart1_mass, self.cart1_size)
        self.cart1_body = pymunk.Body(mass=self.cart1_mass, moment=self.cart1_moment)
        self.cart1_body.position = -.75, self.cart1_size[1] / 2
        self.cart1_shape = pymunk.Poly.create_box(self.cart1_body, self.cart1_size)
        self.cart1_shape.friction = self.ground.friction
        self.space.add(self.cart1_body, self.cart1_shape)

        # create cart2
        self.cart2_mass = 0.5
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
        self.pend1_body.angle = -math.pi / 4
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
        self.pend2_body.angle = -3 * math.pi / 4
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
        self.pend3_body.angle = 0
        self.pend3_body.position = 0, -2 * self.pend1_length * math.sin(self.pend1_body.angle) + self.cart1_size[1] / 2 + self.pend3_length - 0.05
        self.pend3_shape = pymunk.Poly.create_box(self.pend3_body, self.pend3_size)
        self.pend3_shape.filter = fil
        self.space.add(self.pend3_body, self.pend3_shape)
        
        # create joint
        self.joint1 = pymunk.constraints.PivotJoint(self.cart1_body, self.pend1_body, self.cart1_body.position + (0, self.cart1_size[1] / 2))
        self.joint1.collide_bodies = False
        self.space.add(self.joint1)
        
        
        # create joint 2
        self.joint2 = pymunk.constraints.PivotJoint(self.cart2_body, self.pend2_body, self.cart2_body.position + (0, self.cart1_size[1] / 2))
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
        

    def step(self, dt, action_select):
        """
        Take in and apply actions, step pymunk space, output new state variables, reward, and done
        """
        # print(self.action_space[action_select])
        action = self.action_space[action_select]
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

        # self.state = (x1, x1_dot, x2, x2_dot, t1, w1, t2, w2, tp, wp, xp, yp)
        self.state = (x1, x1_dot, x2, x2_dot, t1, w1, t2, w2, tp, wp)
        
        # print('Pend 3 angle: ' ,self.state[8]*180/math.pi)
        
        # Stopping condition (angle of pole 3 is in 60:300, ie. over 60 degrees from upright)
        done = bool(
                tp > math.pi/3
                and tp < 5*math.pi / 2
        )

        # print(done)

        
        # action is the force to two carts, [f1, f2]
        # f1 can be either [1, -1 , 0]


        if not done:
            reward = 1.0
        else:
            reward = 0

        return self.state, reward, done

    def reset(self):
        if self.space:
            del self.space
        self._init_objects()
