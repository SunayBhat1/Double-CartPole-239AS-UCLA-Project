import gym
from carts_poles import CartsPolesEnv
import pyglet
import pymunk
SCREEN_HEIGHT = 700
PPM = 200.0
window = pyglet.window.Window(1000, SCREEN_HEIGHT, vsync=False, caption='Inverted Pendulum Simulator')

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


def draw_ground(offset, ground):
    vertices = [v + (0, ground.radius) for v in (ground.a, ground.b)]

    # convert vertices to pixel coordinates
    points = []
    for v in vertices:
        points.append(int(v[0] * PPM) + offset[0])
        points.append(int(v[1] * PPM) + offset[1])

    data = ('v2i', tuple(points))
    pyglet.graphics.draw(len(vertices), pyglet.gl.GL_LINES, data)


@window.event
def on_draw():
    window.clear()

    
    offset = (500, 250)
    draw_body(offset, env.cart1_body)
    draw_body(offset, env.pend1_body)
    draw_body(offset, env.cart2_body)
    draw_body(offset, env.pend2_body)
    draw_body(offset, env.pend3_body)  
    draw_ground(offset, env.ground)

if __name__ == "__main__":

    global env 
    env = CartsPolesEnv()
    pyglet.clock.schedule_interval(env.step, 0.2, action=(0,1))
    pyglet.app.run()
    #for _ in range(120):
    #    env.step(1)
    #env.reset()




