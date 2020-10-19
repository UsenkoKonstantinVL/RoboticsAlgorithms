import numpy as np
import random


LANDMARKS = [[10, 20], [5, 45], [25, 25], [47, 10], [36, 12]]
WORLD_SIZE = 50

class Robot(object):
    def __init__(self):
        self.x = WORLD_SIZE / 2
        self.y = WORLD_SIZE / 2

        self.weight = 1

    def do_move(self):
        dx = random.normalvariate(0, 2)
        dy = random.normalvariate(0, 2)
        self.x += dx
        self.y += dy

        self.x = self.__normilize(self.x)
        self.y = self.__normilize(self.y)
        return [dx, dy]

    def move(self, move):
        self.x = self.x + move[0] + random.normalvariate(0,  move[0])
        self.y = self.y + move[1] + random.normalvariate(0, move[0])
        
        self.x = self.__normilize(self.x)
        self.y = self.__normilize(self.y)

    def __normilize(self, val):
        val = max(0, val)
        val = min(WORLD_SIZE, val)
        return val

    def do_sense(self):
        sense = list()
        for landmark in LANDMARKS:
            sense.append(self.__calc_dist([self.x, self.y], landmark))
        return sense

    def sense(self, sense):
        self.weight = 1.0
        for i in range(len(LANDMARKS)):
            landmark = LANDMARKS[i]
            est_dist = self.__calc_dist([self.x, self.y], landmark)
            dist = sense[i]
            _weight = self.__calc_prob(np.fabs(dist - est_dist), dist / 5.0)
            self.weight *= _weight

        self.weight = max(self.weight, 0.00000001)

    def __calc_prob(self, val, sigma):
        # sigma = val / 5.0
        p = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1.0/2.0 * np.power(val / sigma, 2))

        return p

    def __calc_dist(self, pos1, pos2):
        return np.linalg.norm(np.array(pos2) - np.array(pos1))

def create_particles(part_count):
    _particles = list()
    for i in range(part_count):
        _particles.append(Robot())
    return _particles


def noise(data):
    if isinstance(data, list):
        for d in data:
            d += random.normalvariate(0, d / 5.0)
    else:
        data += random.normalvariate(0, data / 5.0)
    return data


def resample(particles):
    p3 = []
    N = len(particles)
    index = int(random.random() * N)
    beta = 0
    w = [particle.weight for particle in particles]
    mw = max(w)

    for i in range(N):
        beta += random.random() * 2 * mw
        while beta > w[index]:
            beta -= w[index]
            index = (index + 1)%N
        p3.append(particles[index])
    return p3


def calc_pos(particles):
    sum_w = sum([particle.weight for particle in particles])
    pos_x = 0
    pos_y = 0
    for particle in particles:
        pos_x += particle.x * particle.weight / sum_w
        pos_y += particle.y * particle.weight / sum_w

    return [pos_x, pos_y]


real_robot = Robot()

particles = create_particles(part_count=100)
MOVE_COUNT = 100

for _ in range(MOVE_COUNT):
    move = real_robot.do_move()
    noise_move = noise(move)
    landmarks = real_robot.do_sense()
    noise_landmarks = noise(landmarks)
    for particle in particles:
        particle.move(noise_move)
        particle.sense(noise_landmarks)

    # print([particle.weight for particle in particles])
    f_pos = calc_pos(particles)
    print("Current robot pos: ", real_robot.x, real_robot.y)
    print("Pos estimated by filter: ", f_pos)

    particles = resample(particles)
