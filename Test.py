from gym import Env
from gym.spaces import Discrete, Box #Espacio discreto y espacio caja
import numpy as np
import random
import sys
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import itertools
import subprocess

Res_cpu = 100 #x10^-2
Res_me = 4000000000
eta = -1 #η
theta = 1 #θ
nu_cpu = 0.2 #v

estados = [[],[],[],[]]
pasos = 1000
pasosList = list(range(0, pasos))
train_steps = 50000
action_space_size = 3

# Define el entorno
class NetworkEnv(Env):
    def __init__(self):
        # Acciones disponibles; subir, bajar, mantener
        self.action_space = Discrete(action_space_size)
        # Arreglo porcentaje de recursos asignados y usados
        self.observation_space = Box(low=np.array([0]), high=np.array([Res_cpu]))
        # Establece los recursos asignados y usados
        self.state = 50
        # Duracion
        self.length = pasos

    def step(self, action):
        # Modifica la accion a tomar en positivo y negativo
        take_action = action - 1
        # Aplica la acción
        self.state += take_action
        # Reduce la duracion en 1
        self.length -= 1

        # Calcula la recompensa
        if (self.state >= 45 and self.state <= 55):
            # Agrega violación
            reward = theta
        else:
            # Reinicia violación
            reward = eta

        print(self.state)

        """ # Fallo total CPU
        if self.state[0] >= Res_cpu or self.state[0] <= 0:
            done = True """

        # Duracion completada
        if self.length <= 0:
            done = True
        else:
            done = False

        # Aplica ruido
        #self.state += random.randint(-1,1)
        # Establece un marcador de posicion para la informacion
        info = {}

        # Retorna la informacion del step
        return self.state, reward, done, info

    def render(self, mode="human"):
      estados[0].append(self.state)

      if len(estados[0])==pasos:
          ypoints = np.array(estados[0])
          xpoints = np.array(pasosList)
          plt.xlabel("Pasos")
          plt.ylabel("Recursos asignados")
          #plt.plot(xpoints, ypoints_ar_me, color ='orange', label ='C CPU (Asignados)')
          #plt.plot(xpoints, ypoints_ur_me, color ='g', label ='U CPU (Usados)')
          plt.plot(xpoints, ypoints, color ='r', label ='Ar CPU (Asignados)')
          plt.legend()
          plt.grid()
          plt.show()
          estados[0].clear()
      return

    def reset(self):
        # Reinicia recursos asignados
        self.state = 0
        # Reinicia la duracion
        self.length = pasos
        return self.state

def train(states, actions):
    model = get_model(states, actions)
    model.summary()

    dqn = get_agent(model, actions)

    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=train_steps, visualize=False, verbose=1)

    #scores = dqn.test(env, nb_episodes=100, visualize=False)
    #print(scores.history)
    #print(np.mean(scores.history['episode_reward']))

    dqn.save_weights('dqn_weights.h5f', overwrite=True)

def test(states, actions):
    model = get_model(states, actions)
    dqn = get_agent(model, actions)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.load_weights('dqn_weights.h5f')
    dqn.test(env, nb_episodes=10, visualize=True)

def get_model(states, actions):
    # Crea la red neuronal
    model = Sequential()

    # Aplanar unidades
    #model.add(Flatten(input_shape=(states)))

    # Add a hidden layers with dropout
    model.add(Dense(24, activation="relu", input_shape=states))
    model.add(Dense(24, activation="relu"))
    #model.add(Dropout(0.5))

    # Add an output layer with output units
    model.add(Dense(actions, activation="linear"))

    return model

def get_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Es necesario colocar un argumento:")
        print("Entrenamiento: Train, Ensayos: Test")
    else:
        env = NetworkEnv()
        env.reset()
        states = env.observation_space.shape
        actions = env.action_space.n
        if sys.argv[1] == "Train":
            train(states, actions)
        elif sys.argv[1] == "Test":
            test(states, actions)
        else:
            print("Argumento "+sys.argv[1]+" no es válido")
            print("Entrenamiento: Train, Ensayos: Test")
        #state_space()