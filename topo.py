from mininet.node import CPULimitedHost, RemoteController, OVSSwitch
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.log import setLogLevel, info
from mininet.cli import CLI

class CustomTopo( Topo ):
  # Crea topologia personalizada

  def __init__(self, **opts):
    # Inicializa la topologia
    super(CustomTopo, self).__init__(**opts)

    # Agrega los hosts
    h1 = self.addHost('h1')
    h2 = self.addHost('h2')
    h3 = self.addHost('h3')
    h4 = self.addHost('h4')

    # Agrega los switches
    s1 = self.addSwitch('s1')
    s2 = self.addSwitch('s2')

    # Agrega los enlaces
    self.addLink(s1, s2)

    # s1
    self.addLink(h1, s1)
    self.addLink(h2, s1)
    self.addLink(h3, s1)

    # s2
    self.addLink(h4, s2)

def run():
  # Crea el controlador
  c = RemoteController('c', '127.0.0.1', 6633)

  # Crea la red
  net = Mininet(
    topo=CustomTopo(),
    controller=None,
    host=CPULimitedHost,
    switch=OVSSwitch,
  )

  # Agrega el controlador
  net.addController(c)

  # Inicia la red
  net.start()

  # Ejecuta la aplicacion
  CLI(net)

  # Detiene la red
  net.stop()

# sudo custom/nombre.py:
if __name__ == '__main__':
  setLogLevel('info')
  run()