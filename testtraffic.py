import os

def setReceiver():
    os.system('sudo docker exec -t mn.d2 ./ITGRecv/ITGRecv &')

def sendTraffic():
    #Prueba de tráfico de una cirugía remota
    os.system('sudo docker exec -t mn.d1 ./ITGSend/ITGSend -a 10.0.0.252 -rp 10001 -C 10000 -c 20000 -T UDP -t 10000 -l sender.log -x receiver.log &')
    os.system('sudo docker exec -t mn.d1 ./ITGSend/ITGSend -a 10.0.0.252 -rp 10002 -C 10000 -c 20000 -T UDP -t 10000 -l sender.log -x receiver.log &')

sendTraffic()
