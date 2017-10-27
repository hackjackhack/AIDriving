#!/usr/bin/python

import argparse
import os.path
import random
import socket, sys
import struct
import time
import numpy as np

from keras import backend as KERAS
from keras import optimizers
from keras import utils
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D

# Configuration
ALPHA = 0.0001
GAMMA = 0.99
LOOK_BACK = 4
MAX_STEPS_TO_TRAIN = 2000
LISTEN_PORT=12345

# Constant
ACCELERATE = 0
BRAKE = 1
TURN_LEFT = 2
TURN_RIGHT = 3
NB_ACTION = 4
ACTIONS = [
  'ACCELERATE', 'BRAKE     ', 'LEFT      ', 'RIGHT     '
]

# Misc functions
def isBlack(state):
  ret = True
  for e in np.nditer(state):
    if e > 10:
      ret = False
  return ret;

def toNPArray(buf, width, height):
  arr = np.empty([height, width, 1])
  for y in range(0, height):
    for x in range(0, width):
      green = buf[y * width + x]
      arr[y][x][0] = green
  return arr

def combine(frames, height, width):
  blank = np.zeros([height, width, 1])
  while len(frames) < LOOK_BACK:
    frames.insert(0, blank)
  return np.dstack(frames)

#####################################
#      Deep Network Definition      #
#####################################
def createNN(inputShape, weightFilepath = None):
  imageInput = Input(shape=inputShape, name='imageInput')
  auxiliaryInput = Input(shape=(1,), name='auxiliaryInput')
  conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform')(imageInput)
  conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform')(conv1)
  conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform')(conv2)
  flatten = Flatten()(conv3)
  merge = Concatenate()([flatten, auxiliaryInput])
  dense1 = Dense(512, activation='tanh', kernel_initializer='random_uniform', bias_initializer='random_uniform')(merge)
  dense2 = Dense(NB_ACTION, activation='softmax', kernel_initializer='random_uniform', bias_initializer='random_uniform')(dense1)

  model = Model(inputs=[imageInput, auxiliaryInput], outputs=[dense2])

  if weightFilepath is not None:
      model.load_weights(weightFilepath)

  return model

def createTrainFunction(model):
  # Reference: https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2
  actionProb = model.output
  actionOneHotCode = KERAS.placeholder(shape=(None, NB_ACTION), name="onehot")
  reward = KERAS.placeholder(shape=(None,), name="reward")
  loss = KERAS.mean(-KERAS.log(KERAS.sum(actionProb * actionOneHotCode, axis=1)) * reward)
  adam = optimizers.Adam(lr = ALPHA)
  updates = adam.get_updates(params=model.trainable_weights,
                             loss=loss)
  return KERAS.function(inputs=[model.input[0], model.input[1],
                                actionOneHotCode,
                                reward],
                        outputs=[],
                        updates=updates)


#####################################
#      Reinforcement Learning       #
#####################################
class Player:
  def __init__(self, args):
    self.episode = 0
    self.historyStepCounter = 0
    self.cumulatedReward = 0
    self.memory = [[], [], [], []]
    self.frames = []
    self.height = -1
    self.width = -1
    self.nn = None
    self.episodeStepCounter = 0
    self.lastState = None
    self.lastAction = BRAKE
    self.lastVelocity = 0
    self.args = args
    print self.args
    self.rewardLog = open(self.args['reward_log'], 'w')

  def reset(self):
    self.rewardLog.write('%d,%10.2f\n' % (self.episode, self.cumulatedReward / float(self.episodeStepCounter + 1)))
    self.rewardLog.flush()
    self.episode += 1
    print 'Episode #' + str(self.episode) + ', Memory size = ' + str(len(self.memory[0]))
    if not self.args['inference_only']:
      self.reinforce()
    self.episodeStepCounter = 0
    self.cumulatedReward = 0
    self.frames = []
    blank = np.zeros([self.height, self.width, 1])
    for i in range(0, LOOK_BACK - 1):
      self.frames.append(blank)
    self.lastState = np.zeros([self.height, self.width, LOOK_BACK])
    self.lastAction = ACCELERATE
    self.lastVelocity = 0

  def discountReward(self, r):
    discountedR = []
    total = 0
    for i in reversed(range(len(r))):
      total = total * GAMMA + r[i]
      discountedR.append(total)
    discountedR = np.array(discountedR[::-1]) # Reverse it
    u = discountedR.mean()
    std = discountedR.std() + 0.000000001
    return (discountedR - u) / std

  def reinforce(self):
    if len(self.memory[0]) <= 0:
      return
    actionOneHotCode = utils.to_categorical(self.memory[2], num_classes=NB_ACTION)
    print 'Training...'
    self.trainFunction([self.memory[0], self.memory[1], actionOneHotCode, self.discountReward(self.memory[3])])
    self.memory = [[], [], [], []]

  def sampleAction(self, reward, velocity, newState, isEnd):
    self.cumulatedReward += reward
    self.episodeStepCounter += 1
    self.historyStepCounter += 1

    if isEnd:
      print 'Score = ' + str(self.cumulatedReward / float(self.episodeStepCounter))
      self.reset()
      return BRAKE

    # Decide action by sampling
    probs = self.nn.predict([np.array([newState]), np.array([[velocity]])])[0]
    action = np.random.choice(NB_ACTION, p=probs)
    log = 'lastAction = ' + ACTIONS[self.lastAction]
    log += ', reward = ' + '%7.2f' % (reward)
    log += ', V = ' + '%10.5f' % (velocity)
    log += ', action = ' + ACTIONS[action]
    for i in range(NB_ACTION):
      if i == 0:
        log += ', P=[%.4f' % (probs[i])
      else:
        log += ', %.4f' % (probs[i])
    log += ']'
    print log

    if not self.args['inference_only'] and not isBlack(self.lastState):
      self.memory[0].append(self.lastState)
      self.memory[1].append([self.lastVelocity])
      self.memory[2].append(self.lastAction)
      self.memory[3].append(reward)
      if len(self.memory[0]) > MAX_STEPS_TO_TRAIN:
        self.reinforce()

    self.lastAction = action
    self.lastState = newState
    self.lastVelocity = velocity

    if (not self.args['inference_only']) \
       and ('model_file' in self.args) \
       and self.historyStepCounter % 10000 == 0:
      self.nn.save_weights(self.args['model_file'], overwrite=True)
      print 'model saved'
  
    return action
    
  def readFrame(self, sock):
    header = sock.recv(struct.calcsize('IIffIII'))
    magic, frameCounter, reward, velocity, restart, width, height = struct.unpack('IIffIII', header)
    if magic != 0xdeadbeef:
      print 'Communication error'
      sys.exit(1)
    imageLen = struct.calcsize('%sB' % (width * height))
    image = bytearray()
    while imageLen > 0:
      buf = sock.recv(imageLen)
      imageLen -= len(buf)
      image += buf
    return (reward, velocity, restart, width, height, image)

  def play(self):
    try:
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error, msg:
      sys.stderr.write("[ERROR] %s\n" % msg[1])
      sys.exit(1)
    
    try:
      sock.connect(('localhost', LISTEN_PORT))
    except socket.error, msg:
      sys.stderr.write("[ERROR] %s\n" % msg[1])
      exit(1)
    
    print 'Simulator connected, model file name = ' + self.args['model_file']
    
    combinedReward = 0
    while True:
      reward, velocity, ending, width, height, image = self.readFrame(sock)
      if self.nn is None:
        if os.path.isfile(self.args['model_file']):
          self.nn = createNN((height, width, LOOK_BACK), self.args['model_file'])
        else:
          self.nn = createNN((height, width, LOOK_BACK))
        self.trainFunction = createTrainFunction(self.nn)

        self.height = height
        self.width = width
        self.reset()
    
      self.frames.append(toNPArray(image, width, height))
      combinedReward += reward
      if len(self.frames) >= LOOK_BACK or ending > 0:
        state = combine(self.frames, height, width)
        action = self.sampleAction(combinedReward,
                                   velocity * 1000, # *1000 for prettiness
                                   state,
                                   ending > 0)
        self.frames = []
        combinedReward = 0
      else:
        action = self.lastAction
      sock.send(struct.pack('I', action))
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Argument parser')
  parser.add_argument('-i', '--inference-only', action='store_true')
  parser.add_argument('--reward-log', default='reward.log')
  parser.add_argument('-m', '--model-file', default='model.h5')
  args = parser.parse_args()

  player = Player(vars(args))
  player.play()

