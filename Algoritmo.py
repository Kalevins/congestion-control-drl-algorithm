from gym import Env
from Monitoring import *
import math
import datetime
from gym.spaces import Discrete, Box #Espacio discreto y espacio caja
import numpy as np
import random
import sys
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import MaxBoltzmannQPolicy
from rl.memory import SequentialMemory

# Numero de cirugias remotas simultaneas
numRemoteSurgeries = 3
# Numero de repeticiones
numRepeats=3
# Numero maximo de unidades de CPU (10^-1) y de memoria (10^8)
resources = {'cpu': 100, 'mem': 40}
# η
eta = -1
# θ
theta = 1
# v
#nu_cpu = 0.2
nu_cpu = 5
# Numero inicial de unidades de CPU en el nodo 1 (sender)
numCoresD1=4
# Numero inicial de unidades de CPU en el nodo 2 (receiver)
numCoresD2=1
# Numero de varibles de estado
estados = [[],[],[],[]]
# Numero de pasos para tomar la desición
pasos = 100
pasosList = list(range(0, pasos))
# Numero de pasos de entrenamiento
trainSteps = 100000
# Tamaño del espacio de acciones
actionSpaceSize = 21

# Define el entorno
class NetworkEnv(Env):
    # Condiciones iniciales
    def __init__(self):
        # Acciones disponibles;
        self.action_space = Discrete(actionSpaceSize)
        # Arreglo de recursos asignados y usados
        self.observation_space = Box(low=0, high=resources[sys.argv[1]], shape=(1,5))
        # Establece los recursos asignados y usados
        self.state = [0,0,0,0,0]
        # Duracion
        self.length = pasos
        # Numero Cores D1
        self.numCoresD1 = numCoresD1
        # Numero Cores D2
        self.numCoresD2 = numCoresD2
        # Arreglo latencia obtenida
        self.latency = []

    # Paso a seguir
    def step(self, action):
        # Modifica la accion a tomar en positivo y negativo
        take_action = action - 10
        # Aplica la acción
        self.state[0] += take_action
        # Aplica ruido
        self.state[2] += random.randint(-1,1)
        if(self.state[2]<=0):
            self.state[2] = 1
        # Reduce la duracion en 1
        self.length -= 1

        # Calcula la recompensa
        if(self.state[0] >= resources[sys.argv[1]] or self.state[0] <= 0):
            # Calcula recompensa
            reward = eta*100
        elif (self.state[0] >= self.state[2] + nu_cpu):
            # Agrega violación
            self.state[4] += 0.001
            # Calcula recompensa
            reward = theta*((self.state[2] + nu_cpu)/self.state[0])
        else:
            # Reinicia violación
            self.state[4] = 0.0
            # Calcula recompensa
            reward = eta*np.exp(self.state[4])
            #reward = eta

        # Duracion completada
        if self.length <= 0:
            # En condicciones ejecución
            if sys.argv[1] == "Start":
                # Reasignacion de unidades de CPU
                self.numCoresD2 = int(math.ceil(self.state[0]/10))
            done = True
        else:
            done = False

        # Establece un marcador de posicion para la informacion
        info = {}

        # Retorna la informacion del step
        return self.state, reward, done, info

    # Graficos
    def render(self, mode="human"):
        estados[0].append(self.state[0])
        estados[1].append(self.state[1])
        estados[2].append(self.state[2])
        estados[3].append(self.state[3])

        if len(estados[0])==pasos:
            ypoints_ar = {'cpu': np.array(estados[0]), 'mem': np.array(estados[1])}
            ypoints_ur = {'cpu': np.array(estados[2]), 'mem': np.array(estados[3])}
            ylabels = {'cpu': "CPU [Unidades]", 'mem': "Memoria [MB]"}
            xpoints = np.array(pasosList)
            plt.xlabel("Episodios")
            plt.ylabel(ylabels[sys.argv[1]])
            plt.ylim(0, resources[sys.argv[1]])
            plt.plot(xpoints, ypoints_ar[sys.argv[1]], color = 'r', label = 'Recursos Asignados')
            plt.plot(xpoints, ypoints_ur[sys.argv[1]], color = 'b', label = 'Recursos Usados')
            plt.legend()
            plt.grid()
            plt.show()
            estados[0].clear()
            estados[1].clear()
            estados[2].clear()
            estados[3].clear()
        return

    # Reaudar
    def reset(self):
        # Reinicia recursos asignados
        self.state = get_initial_state(self)

        # Reinicia la duracion
        self.length = pasos

        return self.state

# Obtiene el estado inicial
def get_initial_state(self):
    # En condicciones de ejecución
    if sys.argv[1] == "Start":
        # Inicia la topologia
        StartTopology()
        # Actualiza valores de CPU
        UpdateCPU(self.numCoresD1,self.numCoresD2)
        # Agrega trafico de n cirugias remotas
        AddSurgery(numRemoteSurgeries)
        # Espera
        time.sleep(5)
        # Obtiene el uso de CPU
        ur_cpu=obtainCPUMEM(self.numCoresD1,self.numCoresD2)[1]
        # Obtiene el uso de Memoria
        ur_me=obtainCPUMEM(self.numCoresD1,self.numCoresD2)[2]
        # Espera
        time.sleep(25)
        # Obtiene la latencia
        self.latency.append(ObtainLatency())
        # Guarda la latencia
        np.savetxt('results/latency_'+str(datetime.date.today())+'_'+str(numRemoteSurgeries)+'RS'+'.csv',self.latency,delimiter=',')
        # Detiene la topologia
        ShutDown()
        # Actualiza recursos asignados de CPU
        ar_cpu = self.numCoresD2*10
        # Actualiza recursos asignados de Memoria
        ar_me = 100
    #En entrenamiento y testeo
    else :
        # Asigna valores aleatorios
        ur_cpu = random.randint(0,resources["cpu"])
        ur_me = random.randint(0,resources["mem"])
        ar_cpu = random.randint(0,resources["cpu"])
        ar_me = random.randint(0,resources["mem"])
    # Inicializa violación en 0
    Violation = 0

    return [ar_cpu, ar_me, ur_cpu, ur_me, Violation]

# Entrenamiento
def train(states, actions):
    model = get_model(states, actions)
    model.summary()
    dqn = get_agent(model, actions)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=trainSteps, visualize=False, verbose=1)
    dqn.save_weights('dqn_weights_'+sys.argv[1]+'.h5f', overwrite=True)

# Testeo
def test(states, actions):
    model = get_model(states, actions)
    dqn = get_agent(model, actions)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.load_weights('dqn_weights_'+sys.argv[1]+'.h5f')
    dqn.test(env, nb_episodes=10, visualize=True)

# Ejecución
def start(states, actions):
    model = get_model(states, actions)
    dqn = get_agent(model, actions)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    dqn.load_weights('dqn_weights_'+sys.argv[1]+'.h5f')
    dqn.test(env, nb_episodes=numRepeats, visualize=True)

# Modelo de redes neuronales
def get_model(states, actions):
    # Crea la red neuronal
    model = Sequential()

    # Aplanar unidades
    model.add(Flatten(input_shape=(states)))

    # Agrega capas ocultas
    model.add(Dense(24, activation="relu", input_shape=states))
    model.add(Dense(24, activation="relu"))
    #model.add(Dropout(0.5))

    # Agregar una capa de salida con unidades de salida
    model.add(Dense(actions, activation="linear"))

    return model

# Agente
def get_agent(model, actions):
    # Una combinación de epsilon-greedy y Boltzman q-policy.
    # Epsilon-greedy implementa la política codiciosa de epsilon:
    #  - realiza una acción aleatoria con probabilidad épsilon
    #  - toma la mejor acción actual con prob (1 - epsilon)
    # Boltzman q-policy construye una ley de probabilidad sobre los valores de q y devuelve una acción seleccionada al azar de acuerdo con esta ley.
    policy = MaxBoltzmannQPolicy(eps=0.1)

    # Proporciona una estructura de datos rápida y eficiente en la que podemos almacenar las experiencias del agente.
    memory = SequentialMemory(limit=50000, window_length=1)

    # Crea el agente
    # nb_steps_warmup: Se utilizan para reducir la tasa de aprendizaje con el fin de reducir el impacto de desviar el modelo del aprendizaje en la exposición repentina a nuevos conjuntos de datos.
    # target_model_update: reemplaza la red con una copia nueva sin entrenar de vez en cuando
    # enable_double_dqn: Habilita la doble dqn
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2, enable_double_dqn=True)

    return dqn

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Es necesario indicar los argumento:")
        print("CPU: cpu, Memoria: mem")
        print("Entrenamiento: Train, Ensayos: Test, Ejecutar: Start")
    else:
        if not (sys.argv[1] == "cpu" or sys.argv[1] == "mem"):
            print("Argumento "+sys.argv[1]+" no es válido.")
            print("CPU: cpu, Memoria: mem")
        elif not (sys.argv[2] == "Train" or sys.argv[2] == "Test" or sys.argv[2] == "Start"):
            print("Argumento "+sys.argv[2]+" no es válido.")
            print("Entrenamiento: Train, Ensayos: Test, Ejecutar: Start")
        else:
            env = NetworkEnv()
            env.reset()
            states = env.observation_space.shape
            actions = env.action_space.n
            if sys.argv[2] == "Train":
                train(states, actions)
            elif sys.argv[2] == "Test":
                test(states, actions)
            elif sys.argv[2] == "Start":
                start(states, actions)