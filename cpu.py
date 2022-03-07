import numpy as np
import subprocess

def getMetrics():
    cmd="sudo docker exec -it mn.d1 top -n 1| grep -E '(PID|ITGSend)' | grep -v '%CPU' | awk '{print $10}'"
    cmd2="sudo docker exec -it mn.d1 top -n 1| grep -E '(PID|ITGSend)' | grep -v '%MEM' | awk '{print $11}'"
    ps=subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output=ps.communicate()[0].decode('utf-8')
    ps2=subprocess.Popen(cmd2,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output2=ps2.communicate()[0].decode('utf-8')
    cmd3="sudo docker exec -it mn.d2 /home/D-ITG-2.8.1-r1023/bin/ITGDec /home/D-ITG-2.8.1-r1023/bin/receiver.log | awk 'NR==12 {print $4}'"
    ps3=subprocess.Popen(cmd3,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output3=ps3.communicate()[0].decode('utf-8')
    latency=float(output3)*1000
    arrayCPU = np.array([float(output.rstrip("\n")),float(output2.rstrip("\n")),latency])
    
    return arrayCPU

y=getMetrics()
print (y)