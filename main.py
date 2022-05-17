#!/usr/bin/python
"""
Mininet
"""
import os
import subprocess
import numpy as np
import time
import statistics

from mininet.net import Containernet
from mininet.node import Controller
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
setLogLevel('info')


def StartTopology():
  """
  Function that creates and initializes the network topology
  """
  info('****************************\n')
  info('*** Start Topology ***\n')
  net = Containernet(controller=Controller)
  info('*** Adding controller\n')
  net.addController('c0')
  info('*** Adding docker containers\n')
  d1 = net.addDocker('d1', ip='10.0.0.251', dimage="test:latest",mem_limit="512m")
  d2 = net.addDocker('d2', ip='10.0.0.252', dimage="test:latest",mem_limit="512m")
  info('*** Creating links\n')
  net.addLink(d1, d2,delay='1ms')
  #net.addLink(d1, d2, cls=TCLink, delay='10ms')
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

def UpdateCPU(numCoresD1,numCoresD2):
  """
  Function that allocates and/or modifies the CPU capacity of the nodes  """
  info('****************************\n')
  info('*** Update CPUs ***\n')
  os.system("sudo docker update --cpuset-cpus='"+",".join(str(e) for e in list(range(numCoresD1)))+"' mn.d1")
  os.system("sudo docker update --cpuset-cpus='"+",".join(str(e) for e in list(range(numCoresD1, numCoresD1+numCoresD2)))+"' mn.d2")

def AddSurgery(numSurgeries):
  """
  Function that emulates the traffic of remote surgery
  """
  info('****************************\n')
  info('*** Add Remote Surgery traffic ***\n')
  ports = 10001
  for num in range(numSurgeries):
    info('*** Add one\n')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp '+str((num*5)+(ports))+' -C 3 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp '+str((num*5)+(ports+1))+' -C 10000 -c 20000 -T UDP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp '+str((num*5)+(ports+2))+' -C 49 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp '+str((num*5)+(ports+3))+' -C 375 -c 512 -T TCP -t 10000 -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend -a 10.0.0.252 -rp '+str((num*5)+(ports+4))+' -C 32 -c 1568 -T UDP -t 10000 -x receiver.log &')

def obtainCPUMEM(numCoresD1,numCoresD2):
  """
  Function that obtains CPU and Memory metrics
  """
  info('****************************\n')
  info('*** Read Data ***\n')
  #CPU d1
  cmdCPUd1="sudo docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' mn.d1 | grep -v 'CPU' | awk '{print $2}'| sed 's/.$//'"
  psCPUd1=subprocess.Popen(cmdCPUd1,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  outputCPUd1=psCPUd1.communicate()[0].decode('utf-8')
  CPUd1=((numCoresD1*float(outputCPUd1))/100)*10
  #MEM d1
  cmdMEMd1="sudo docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemPerc}}' mn.d1  | awk '{print $3}' | sed 's/.$//'"
  psMEMd1=subprocess.Popen(cmdMEMd1,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  outputMEMd1=psMEMd1.communicate()[0].decode('utf-8')
  MEMd1=(float(outputMEMd1)*512)/100
  #CPU d2
  cmdCPUd2="sudo docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' mn.d2 | grep -v 'CPU' | awk '{print $2}'| sed 's/.$//'"
  psCPUd2=subprocess.Popen(cmdCPUd2,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  outputCPUd2=psCPUd2.communicate()[0].decode('utf-8')
  CPUd2=((numCoresD2*float(outputCPUd2))/100)*10
  #MEM d2
  cmdMEMd2="sudo docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemPerc}}' mn.d2  | awk '{print $3}' | sed 's/.$//'"
  psMEMd2=subprocess.Popen(cmdMEMd2,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  outputMEMd2=psMEMd2.communicate()[0].decode('utf-8')
  MEMd2=(float(outputMEMd2)*512)/100
  #Array
  arrayCPU=np.array([CPUd1,CPUd2,MEMd1,MEMd2])
  arrayCPU=np.around(arrayCPU)
  arrayCPU=arrayCPU.astype(int)
  return arrayCPU

def ObtainLatency():
  #Latency of tests
  latencytests="sudo docker exec -t mn.d2 ./ITGDec receiver.log | tail -n 10 | awk 'NR==1{print $4}'"
  pslatencytests=subprocess.Popen(latencytests,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  outputLatencytests=pslatencytests.communicate()[0].decode('utf-8')
  lat=float(outputLatencytests)*1000
  return lat

def ShutDown():
  """
  Function that turns off the topology
  """
  info('****************************\n')
  info('*** Shutdown Topology ***\n')
  os.system("sudo mn -c")

""" if __name__ == "__main__":
  numCoresD1 = 2
  numCoresD2 = 2
  latency=[]
  StartTopology()
  UpdateCPU()
  AddSurgery(3)
  time.sleep(5)
  print(obtainCPUMEM())
  time.sleep(25)
  latency.append(ObtainLatency())
  #ShutDown()

  mean = statistics.mean(latency)
  print(mean) """