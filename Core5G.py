#!/usr/bin/python
"""
This is the most simple example to showcase Containernet.
"""
from mininet.net import Containernet
from mininet.node import Controller
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
setLogLevel('info')

net = Containernet(controller=Controller)
info('*** Adding controller\n')
net.addController('c0')
info('*** Adding docker containers\n')
d1 = net.addDocker('d1', ip='10.0.0.251', dimage="test:latest",mem_limit="512m")
d2 = net.addDocker('d2', ip='10.0.0.252', dimage="test:latest",mem_limit="512m")
info('*** Adding switches\n')
s1 = net.addSwitch('s1')
info('*** Creating links\n')
net.addLink(d1, d2, cls=TCLink, delay='10ms')
info('*** Starting network\n')
net.start()
info('*** Testing connectivity\n')
net.ping([d1, d2])
info('*** Starting Commands\n')
info('*** Server Start\n')
d2.cmd('/home/D-ITG-2.8.1-r1023/src/ITGRecv/ITGRecv &')
#info('*** Send Traffic\n')
#d1.cmd('/home/D-ITG-2.8.1-r1023/src/ITGSend/ITGSend -a 10.0.0.252 -rp 10001 -C 98 -c 512 -T UDP -t 60000 -l sender.log -x receiver.log')
info('*** Running CLI\n')
CLI(net)
info('*** Stopping network')
net.stop()