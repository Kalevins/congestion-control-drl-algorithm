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

from tensorflow.python.eager.context import PhysicalDevice
from tensorflow.python.keras.callbacks import TensorBoard

Res_cpu = 2.000
Res_me = 4000000000
eta = -1 #η
theta = 1 #θ
nu = 0.8 #v
estados = [[],[]]
pasos = 100
pasosList = list(range(0, pasos))

class NetworkEnv(Env): #Define el entorno
    def __init__(self):
        #Acciones disponibles; subir, bajar, mantener
        self.action_space = Discrete(3)
        #Arreglo porcentaje de recursos asignados
        self.observation_space = Box(low=0, high=100, dtype=float, shape=(1,8))
        #Establece los recursos asignados
        self.state = np.array([0,0,0,0,0,0,0,0], dtype=float)
        #Duracion
        self.length = pasos

    def step(self, action):
        # Aplica la accion
        # 0 -1 = -1 porcentaje de recurso
        # 1 -1 = 0
        # 2 -1 = 1 porcentaje de recurso
        self.state = get_state((self.state[4] + ((action - 1)*0.001)), self.state[5], self.state[6], self.state[7])
        # Reduce la duracion en 1
        self.length -= 1

        # Calcula la recompensa
        if self.state[6] > (self.state[4]*nu):
            reward = eta
        else:
            reward = theta 

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
        estados[0].append(self.state[0])
        estados[1].append(self.state[2])

        if len(estados[0])==100:
            y1points = np.array(estados[0])
            y2points = np.array(estados[1])
            xpoints = np.array(pasosList)
            plt.xlabel("Pasos")
            plt.ylabel("Recursos asignados")
            plt.plot(xpoints, y1points, color ='orange', label ='C CPU (Asignados)')
            plt.plot(xpoints, y2points, color ='g', label ='U CPU (Usados)')
            plt.legend()
            plt.grid()
            plt.show()
            estados[0].clear()
            estados[1].clear()
        return

    def reset(self):
        # Reinicia recursos asignados
        self.state = get_initial_state()
        #self.state = [0,0,random.randint(0, 100),0]
        # Reinicia la duracion
        self.length = pasos
        return self.state

def get_initial_state():
    #data = monitoring("0","0")
    data = randomValues()
    C_cpu = data.ar_cpu / Res_cpu
    C_me = data.ar_me / Res_me
    U_cpu = data.ur_cpu / data.ar_cpu
    U_me = data.ur_me / data.ar_me
    #I_cpu = data.ur_cpu / data.ar_cpu
    #I_me = data.ur_me / data.ar_me

    return [C_cpu, C_me, U_cpu, U_me, data.ar_cpu, data.ar_me, data.ur_cpu, data.ur_me]

def get_state(ar_cpu, ar_me, ur_cpu, ur_me):
    C_cpu = ar_cpu / Res_cpu
    C_me = ar_me / Res_me
    U_cpu = ur_cpu / ar_cpu
    U_me = ur_me / ar_me
    #I_cpu = ur_cpu / ar_cpu
    #I_me = ur_me / ar_me

    return [C_cpu, C_me, U_cpu, U_me, ar_cpu, ar_me, ur_cpu, ur_me]

def monitoring (usageCPUold, usageMemoryold):
    podsData = subprocess.check_output(["kubectl","get","pod","-o","wide"]).decode('ascii').split()
    ranName = podsData[11]
    ipRan = podsData[16]
    upfName = podsData[20]
    ipUpf = podsData[25]

    try:
        limitCPU = eval(subprocess.check_output(["kubectl","get","pod",upfName,"-o","jsonpath='{.spec.containers[0].resources.limits.cpu}'"]).decode('ascii').replace("i",""))
        limitMemory = eval(subprocess.check_output(["kubectl","get","pod",upfName,"-o","jsonpath='{.spec.containers[0].resources.limits.memory}'"]).decode('ascii').replace("i",""))

        latencyData = subprocess.check_output(["kubectl","exec",ranName,"--","ping",ipUpf,"-c","1"]).decode('ascii').split()
        latency = latencyData[12].replace("time=","")
    except:
        podsData = subprocess.check_output(["kubectl","get","pod","-o","wide"]).decode('ascii').split()
        ranName = podsData[11]
        ipRan = podsData[16]

        if (len(podsData) > 35 and podsData[20] == upfName):
            upfName = podsData[29]
            ipUpf = podsData[34]
        else:
            upfName = podsData[20]
            ipUpf = podsData[25]

    try:
        usageData = subprocess.check_output(["kubectl","top","pod",upfName,"--sort-by=cpu"]).decode('ascii').split()
        usageCPU = usageData[4].replace("i","")
        usageMemory = usageData[5].replace("i","")
    except:
        usageCPU = usageCPUold
        usageMemory = usageMemoryold

    state_space = [latency, usageCPU, usageMemory, limitCPU, limitMemory]
    state_space_num = []

    for var in state_space:
        if (var.find("m") != -1):
            state_space_num.append(float(var.replace("m","")) / 1000)
        elif (var.find("K") != -1):
            state_space_num.append(float(var.replace("K","")) * 1000)
        elif (var.find("M") != -1):
            state_space_num.append(float(var.replace("M","")) * 1000000)
        elif (var.find("G") != -1):
            state_space_num.append(float(var.replace("G","")) * 1000000000)
        else:
            state_space_num.append(float(var))

    class data:
        latency = int(state_space_num[0] * 1000)
        ar_cpu = int(state_space_num[3] * 1000)
        ar_me = int(state_space_num[4] * 1000)
        ur_cpu = int(state_space_num[1] * 1000)
        ur_me = int(state_space_num[2] * 1000)

    return data()

def randomValues():
    class data:
        latency = random.uniform(0.000, 2.000)
        ar_cpu = random.uniform(0.000, Res_cpu)
        ar_me = random.randint(0, Res_me)
        ur_cpu = random.uniform(0.000, ar_cpu)
        ur_me = random.randint(0, ar_me)

    return data()

def main1():
    model = get_model(states, actions)
    model.summary()

    dqn = get_agent(model, actions)

    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

    #scores = dqn.test(env, nb_episodes=100, visualize=False)
    #print(scores.history)
    #print(np.mean(scores.history['episode_reward']))

    dqn.test(env, nb_episodes=10, visualize=False)
    dqn.save_weights('dqn_weights.h5f', overwrite=True)

def main2():
    model = get_model(states, actions)
    dqn = get_agent(model, actions)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.load_weights('dqn_weights.h5f')
    dqn.test(env, nb_episodes=10, visualize=True)

def get_model(states, actions):
    # Create a neural network
    model = Sequential()

    # Flatten units
    model.add(Flatten(input_shape=(states)))

    # Add a hidden layers with dropout
    model.add(Dense(24, activation="relu", input_shape=states))
    #model.add(Dropout(0.5))
    model.add(Dense(24, activation="relu"))
    #model.add(Dropout(0.5))


    # Add an output layer with output units
    #model.add(Dense(actions, activation="softmax")
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
        print("Entrenamiento: Train")
        print("Ensayos: Test")
    else:
        env = NetworkEnv()
        env.observation_space.sample()
        env.reset()
        episodes = 10
        for episode in range(1, episodes+1):
            state = env.reset()
            done = False
            score = 0

            while not done:
                #env.render()
                action = env.action_space.sample()
                n_state, reward, done, info = env.step(action)
                score+=reward
            print('Episode:{} Score:{}'.format(episode, score))
        #env.close()

        states = env.observation_space.shape
        actions = env.action_space.n
        if sys.argv[1] == "Train":
            main1()
        elif sys.argv[1] == "Test":
            main2()
        else:
            print("Opción inválida")
        #state_space()
