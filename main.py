#!/usr/bin/python
"""
This is the most simple example to showcase Containernet.
"""
import os
import subprocess
import numpy as np

from mininet.net import Containernet
from mininet.node import Controller
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
setLogLevel('info')


def StartTopology():
    net = Containernet(controller=Controller)
    info('*** Adding controller\n')
    net.addController('c0')
    info('*** Adding docker containers\n')
    d1 = net.addDocker('d1', ip='10.0.0.251', dimage="test:latest",mem_limit="512m")
    d2 = net.addDocker('d2', ip='10.0.0.252', dimage="test:latest",mem_limit="512m")
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
    #info('*** Stopping network')
    #net.stop()

def UpdateCPU(a,b):
    if a==8 and b==4:
        os.system("sudo docker update --cpuset-cpus='0,1,2,3,4,5,6,7' mn.d1")
        os.system("sudo docker update --cpuset-cpus='8,9,10,11' mn.d2")
    elif a==8 and b==2:
        os.system("sudo docker update --cpuset-cpus='0,1,2,3,4,5,6,7' mn.d1")
        os.system("sudo docker update --cpuset-cpus='8,9' mn.d2")
    elif a==8 and b==1:
        os.system("sudo docker update --cpuset-cpus='0,1,2,3,4,5,6,7' mn.d1")
        os.system("sudo docker update --cpuset-cpus='8' mn.d2")
    return a,b

def sendTraffic():
    #Prueba de tráfico de una cirugía remota
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10001 -C 10000 -c 20000 -T UDP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10002 -C 3 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10003 -C 49 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10004 -C 375 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10005 -C 32 -c 1568 -T UDP -t 10000 -x receiver.log &')

def addSurgery():
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10006 -C 10000 -c 20000 -T UDP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10007 -C 3 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10008 -C 49 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10009 -C 375 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp 10010 -C 32 -c 1568 -T UDP -t 10000 -x receiver.log &')

def readData(x,y):
    cmd="sudo docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' mn.d1 | grep -v 'CPU' | awk '{print $2}'| sed 's/.$//'"
    cmd2="sudo docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemPerc}}' mn.d1  | awk '{print $3}' | sed 's/.$//'"
    cmd3="sudo docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' mn.d2 | grep -v 'CPU' | awk '{print $2}'| sed 's/.$//'"
    cmd4="sudo docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemPerc}}' mn.d2  | awk '{print $3}' | sed 's/.$//'"
    ps=subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output=ps.communicate()[0].decode('utf-8')
    ps2=subprocess.Popen(cmd2,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output2=ps2.communicate()[0].decode('utf-8')
    ps3=subprocess.Popen(cmd3,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output3=ps3.communicate()[0].decode('utf-8')
    ps4=subprocess.Popen(cmd4,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output4=ps4.communicate()[0].decode('utf-8')
    CPUd1=(x*float(output))/100
    CPUd2=(y*float(output3))/100
    MEMd1=(float(output2)*512)/100
    MEMd2=(float(output4)*512)/100
    arrayCPU=np.array([CPUd1,CPUd2,MEMd1,MEMd2])
    print (arrayCPU)

def ShutDown():
    os.system("sudo mn -c")


#StartTopology()
x,y=UpdateCPU(8,4)
print(x,y)
sendTraffic()
readData(x,y)

