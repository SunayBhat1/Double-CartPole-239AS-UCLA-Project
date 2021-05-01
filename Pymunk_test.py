import csv
import math
import pymunk.pygame_util
import pyglet
import pymunk
import pymunk.constraints

SCREEN_HEIGHT = 700
window = pyglet.window.Window(1000, SCREEN_HEIGHT, vsync=False, caption='Inverted Pendulum Simulator')

# setup the space
space = pymunk.Space()
space.gravity = 0, -9.8

fil = pymunk.ShapeFilter(group=1)

# ground
ground = pymunk.Segment(space.static_body, (-4, -0.1), (4, -0.1), 0.1)
ground.friction = 0.1
ground.filter = fil
space.add(ground)

# cart 1
cart1_mass = 0.5
cart1_size = 0.3, 0.2
cart1_moment = pymunk.moment_for_box(cart1_mass, cart1_size)
cart1_body = pymunk.Body(mass=cart1_mass, moment=cart1_moment)
cart1_body.position = -.75, cart1_size[1] / 2
cart1_shape = pymunk.Poly.create_box(cart1_body, cart1_size)
cart1_shape.friction = ground.friction
space.add(cart1_body, cart1_shape)

# cart 2
cart2_mass = 0.5
cart2_size = 0.3, 0.2
cart2_moment = pymunk.moment_for_box(cart2_mass, cart2_size)
cart2_body = pymunk.Body(mass=cart1_mass, moment=cart2_moment)
cart2_body.position = .75, cart1_size[1] / 2
cart2_shape = pymunk.Poly.create_box(cart2_body, cart2_size)
cart2_shape.friction = ground.friction
space.add(cart2_body, cart2_shape)


# pendulum
pend1_length = 0.6  # to center of mass
pend1_size = 0.01, pend1_length * 2  # to get CoM at 0.6 m
pend1_mass = 0.2
pend1_moment = pymunk.moment_for_box(pend1_mass, pend1_size)
pend1_body = pymunk.Body(mass=pend1_mass, moment=pend1_moment)
pend1_body.angle=-math.pi/4
pend1_body.position = cart1_body.position[0]+pend1_length*math.cos(pend1_body.angle), cart1_body.position[1] + cart1_size[1] / 2 -pend1_length*math.sin(pend1_body.angle)
pend1_shape = pymunk.Poly.create_box(pend1_body, pend1_size)
pend1_shape.filter = fil
space.add(pend1_body, pend1_shape)

# pendulum 2
pend2_length = 0.6  # to center of mass
pend2_size = 0.01, pend2_length * 2  # to get CoM at 0.6 m
pend2_mass = .2
pend2_moment = pymunk.moment_for_box(pend2_mass, pend2_size)
pend2_body = pymunk.Body(mass=pend2_mass, moment=pend2_moment)
pend2_body.angle=-3*math.pi/4
pend2_body.position = cart2_body.position[0]+pend2_length*math.cos(pend2_body.angle), cart2_body.position[1] + cart2_size[1] / 2 - pend2_length*math.sin(pend2_body.angle)
pend2_shape = pymunk.Poly.create_box(pend2_body, pend2_size)
pend2_shape.filter = fil
space.add(pend2_body, pend2_shape)

# joint
joint1 = pymunk.constraints.PivotJoint(cart1_body, pend1_body, cart1_body.position + (0, cart1_size[1] / 2))
joint1.collide_bodies = False
space.add(joint1)


# joint 2
joint2 = pymunk.constraints.PivotJoint(cart2_body, pend2_body, cart2_body.position + (0, cart1_size[1] / 2))
joint2.collide_bodies = False
space.add(joint2)

# joint 3
joint3 = pymunk.constraints.PivotJoint(pend1_body, pend2_body, (0,-2*pend1_length*math.sin(pend1_body.angle)+cart1_size[1]/2))
joint3.collide_bodies = True
space.add(joint3)


print(f"cart mass = {cart1_body.mass:0.1f} kg")
print(f"pendulum mass = {pend1_body.mass:0.1f} kg, pendulum moment = {pend1_body.moment:0.3f} kg*m^2")

# K gain matrix and Nbar found from modelling via Jupyter
K = [-57.38901804, -36.24133932, 118.51380879, 28.97241832]
Nbar = -57.25

# simulation stuff
force = 0.0
MAX_FORCE = 25
DT = 1 / 60.0
ref = 0.0

# drawing stuff
# pixels per meter
PPM = 200.0

color = (200, 200, 200, 200)
label_x = pyglet.text.Label(text='', font_size=18, color=color, x=10, y=SCREEN_HEIGHT - 28)
label_ang = pyglet.text.Label(text='', font_size=18, color=color, x=10, y=SCREEN_HEIGHT - 58)
label_force = pyglet.text.Label(text='', font_size=18, color=color, x=10, y=SCREEN_HEIGHT - 88)

labels = [label_x, label_ang, label_force]

# data recorder so we can compare our results to our predictions
f = open('data/invpend.csv', 'w')
out = csv.writer(f)
out.writerow(['time', 'x', 'theta'])
currtime = 0.0
record_data = False


def draw_body(offset, body):
    for shape in body.shapes:
        if isinstance(shape, pymunk.Circle):
            # TODO
            pass
        elif isinstance(shape, pymunk.Poly):
            # get vertices in world coordinates
            vertices = [v.rotated(body.angle) + body.position for v in shape.get_vertices()]

            # convert vertices to pixel coordinates
            points = []
            for v in vertices:
                points.append(int(v[0] * PPM) + offset[0])
                points.append(int(v[1] * PPM) + offset[1])

            data = ('v2i', tuple(points))
            pyglet.graphics.draw(len(vertices), pyglet.gl.GL_LINE_LOOP, data)


def draw_ground(offset):
    vertices = [v + (0, ground.radius) for v in (ground.a, ground.b)]

    # convert vertices to pixel coordinates
    points = []
    for v in vertices:
        points.append(int(v[0] * PPM) + offset[0])
        points.append(int(v[1] * PPM) + offset[1])

    data = ('v2i', tuple(points))
    pyglet.graphics.draw(len(vertices), pyglet.gl.GL_LINES, data)

def draw_point(offset,point):
    point[0]=(int(point[0] * PPM) + offset[0])-1
    point[1]=(int(point[1] * PPM) + offset[1])-1
    points=[point[0],point[1],point[0]+2,point[1],point[0]+2,point[1]+2,point[0],point[1]+2]
    pyglet.graphics.draw(4, pyglet.gl.GL_LINE_LOOP,('v2i',tuple(points)))

@window.event
def on_draw():
    window.clear()

    # center view x around 0
    offset = (500, 250)
    draw_body(offset, cart1_body)
    draw_body(offset, pend1_body)
    draw_body(offset, cart2_body)
    draw_body(offset, pend2_body)
    draw_point(offset,[0,-2*pend1_length*math.sin(pend1_body.angle)+cart1_size[1]/2])
    draw_ground(offset)

    for label in labels:
        label.draw()


def simulate(_):
    # ensure we get a consistent simulation step - ignore the input dt value
    dt = DT

    # simulate the world
    # NOTE: using substeps will mess up gains
    space.step(dt)

    # populate the current state
    posx = cart1_body.position[0]
    velx = cart1_body.velocity[0]
    ang = pend1_body.angle
    angv = pend1_body.angular_velocity

    # dump our data so we can plot
    if record_data:
        global currtime
        out.writerow([f"{currtime:0.4f}", f"{posx:0.3f}", f"{ang:0.3f}"])
        currtime += dt

    # calculate our gain based on the current state
    gain = K[0] * posx + K[1] * velx + K[2] * ang + K[3] * angv

    # calculate the force required
    global force
    force = ref * Nbar - gain

    # kill our motors if we go past our linearized acceptable angles
    if math.fabs(pend1_body.angle) > 0.35:
        force = 0.0

    # cap our maximum force so it doesn't go crazy
    if math.fabs(force) > MAX_FORCE:
        force = math.copysign(MAX_FORCE, force)

    # apply force to cart center of mass
    cart1_body.apply_force_at_local_point((force, 0.0), (0, 0))


# function to store the current state to draw on screen
def update_state_label(_):
    label_x.text = f'Cart X: {cart1_body.position[0]:0.3f} m'
    label_ang.text = f'Pendulum Angle: {pend1_body.angle:0.3f} radians'
    label_force.text = f'Force: {force:0.1f} newtons'


def update_reference(_, newref):
    global ref
    ref = newref


# callback for simulation
pyglet.clock.schedule_interval(simulate, .1)
# pyglet.clock.schedule_interval(update_state_label, 0.25)

# # schedule some small movements by updating our reference
# pyglet.clock.schedule_once(update_reference, 2, 0.2)
# pyglet.clock.schedule_once(update_reference, 7, 0.6)
# pyglet.clock.schedule_once(update_reference, 12, 0.2)
# pyglet.clock.schedule_once(update_reference, 17, 0.0)

pyglet.app.run()

# close the output file
f.close()